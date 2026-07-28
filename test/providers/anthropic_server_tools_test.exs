defmodule ReqLLM.Providers.AnthropicServerToolsTest do
  @moduledoc """
  Server-tool support for the Anthropic provider: content blocks produced by
  API-executed tools (`server_tool_use`, `*_tool_result`) round-trip through
  decode → context → encode, `pause_turn` remains detectable as a finish
  reason, and the `code_execution` server tool can be enabled.
  """
  use ExUnit.Case, async: true

  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Providers.Anthropic

  # Shaped like a real web_search response (captured live): the tool-use and
  # result blocks precede the text that cites them.
  defp server_tool_response_data(stop_reason) do
    %{
      "id" => "msg_test",
      "type" => "message",
      "role" => "assistant",
      "model" => "claude-sonnet-4-6",
      "content" => [
        %{
          "type" => "server_tool_use",
          "id" => "srvtoolu_1",
          "name" => "web_search",
          "input" => %{"query" => "example"}
        },
        %{
          "type" => "web_search_tool_result",
          "tool_use_id" => "srvtoolu_1",
          "content" => [
            %{
              "type" => "web_search_result",
              "title" => "Example",
              "url" => "https://example.com",
              "encrypted_content" => "OPAQUE"
            }
          ]
        },
        %{"type" => "text", "text" => "Here is what I found."}
      ],
      "stop_reason" => stop_reason,
      "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
    }
  end

  defp model do
    {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-6")
    model
  end

  describe "non-streaming decode" do
    test "server-tool blocks become ordered provider_block parts" do
      {:ok, response} =
        Anthropic.Response.decode_response(server_tool_response_data("end_turn"), model())

      assert [
               %ContentPart{
                 type: :provider_block,
                 data: %{"type" => "server_tool_use"} = use_block
               },
               %ContentPart{
                 type: :provider_block,
                 data: %{"type" => "web_search_tool_result"} = result_block
               },
               %ContentPart{type: :text, text: "Here is what I found."}
             ] = response.message.content

      assert use_block["input"] == %{"query" => "example"}
      assert get_in(result_block, ["content", Access.at(0), "encrypted_content"]) == "OPAQUE"
    end

    test "pause_turn normalizes to :incomplete with the raw stop_reason kept" do
      {:ok, response} =
        Anthropic.Response.decode_response(server_tool_response_data("pause_turn"), model())

      # :incomplete is the normalized form; the raw stop_reason in
      # provider_meta is what lets callers tell a resumable pause apart.
      assert response.finish_reason == :incomplete
      assert response.provider_meta["stop_reason"] == "pause_turn"
    end
  end

  describe "round-trip encode" do
    test "provider_block parts re-encode verbatim into the next request" do
      {:ok, response} =
        Anthropic.Response.decode_response(server_tool_response_data("pause_turn"), model())

      context = %ReqLLM.Context{messages: [response.message]}
      body = Anthropic.Context.encode_request(context, model())

      assert [%{role: "assistant", content: blocks}] = body.messages

      assert [
               %{"type" => "server_tool_use", "id" => "srvtoolu_1"},
               %{"type" => "web_search_tool_result", "tool_use_id" => "srvtoolu_1"},
               %{type: "text", text: "Here is what I found."}
             ] = blocks

      # encrypted_content must survive untouched
      assert get_in(Enum.at(blocks, 1), ["content", Access.at(0), "encrypted_content"]) ==
               "OPAQUE"
    end

    test "another provider's blocks are not sent to Anthropic" do
      foreign = ContentPart.provider_block(%{"type" => "web_search_call"}, :openai)

      message = %ReqLLM.Message{
        role: :assistant,
        content: [foreign, ContentPart.text("hi")],
        metadata: %{}
      }

      body = Anthropic.Context.encode_request(%ReqLLM.Context{messages: [message]}, model())

      assert [%{role: "assistant", content: "hi"}] = body.messages
    end
  end

  describe "streaming decode" do
    test "server_tool_use accumulates its input fragments and emits one completed block" do
      events = [
        %{
          data: %{
            "type" => "content_block_start",
            "index" => 0,
            "content_block" => %{
              "type" => "server_tool_use",
              "id" => "srvtoolu_1",
              "name" => "web_search",
              "input" => %{}
            }
          }
        },
        %{
          data: %{
            "type" => "content_block_delta",
            "index" => 0,
            "delta" => %{"type" => "input_json_delta", "partial_json" => ~s({"query":)}
          }
        },
        %{
          data: %{
            "type" => "content_block_delta",
            "index" => 0,
            "delta" => %{"type" => "input_json_delta", "partial_json" => ~s("example"})}
          }
        },
        %{data: %{"type" => "content_block_stop", "index" => 0}}
      ]

      {chunks, _state} =
        Enum.reduce(events, {[], Anthropic.init_stream_state(model())}, fn event, {acc, state} ->
          {event_chunks, next_state} = Anthropic.decode_stream_event(event, model(), state)
          {acc ++ event_chunks, next_state}
        end)

      assert [
               %ReqLLM.StreamChunk{type: :meta, metadata: %{server_tool_started: "web_search"}},
               %ReqLLM.StreamChunk{type: :meta, metadata: %{provider_block: block}}
             ] = chunks

      assert block["type"] == "server_tool_use"
      assert block["input"] == %{"query" => "example"}
    end

    test "result blocks arrive complete and emit immediately" do
      event = %{
        data: %{
          "type" => "content_block_start",
          "index" => 1,
          "content_block" => %{
            "type" => "web_search_tool_result",
            "tool_use_id" => "srvtoolu_1",
            "content" => []
          }
        }
      }

      {chunks, _state} =
        Anthropic.decode_stream_event(event, model(), Anthropic.init_stream_state(model()))

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: %{provider_block: block}}] = chunks
      assert block["type"] == "web_search_tool_result"
    end

    test "streaming pause_turn normalizes to :incomplete with the raw stop_reason alongside" do
      event = %{
        data: %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => "pause_turn"},
          "usage" => %{}
        }
      }

      {chunks, _state} =
        Anthropic.decode_stream_event(event, model(), Anthropic.init_stream_state(model()))

      assert Enum.any?(
               chunks,
               &match?(
                 %{metadata: %{finish_reason: :incomplete, stop_reason: "pause_turn"}},
                 &1
               )
             )
    end
  end

  describe "streamed response materialization (ResponseBuilder)" do
    test "provider blocks and the raw stop_reason survive into the built response" do
      chunks = [
        ReqLLM.StreamChunk.meta(%{
          provider_block: %{"type" => "server_tool_use", "id" => "srvtoolu_1"},
          provider: :anthropic
        }),
        ReqLLM.StreamChunk.meta(%{
          provider_block: %{"type" => "web_search_tool_result", "tool_use_id" => "srvtoolu_1"},
          provider: :anthropic
        }),
        ReqLLM.StreamChunk.text("Found it."),
        ReqLLM.StreamChunk.meta(%{
          finish_reason: :incomplete,
          stop_reason: "pause_turn",
          terminal?: true
        })
      ]

      {:ok, response} =
        ReqLLM.Provider.Defaults.ResponseBuilder.build_response(
          chunks,
          %{finish_reason: :incomplete},
          context: %ReqLLM.Context{messages: []},
          model: model()
        )

      assert [
               %ContentPart{type: :provider_block, data: %{"type" => "server_tool_use"}},
               %ContentPart{type: :provider_block, data: %{"type" => "web_search_tool_result"}},
               %ContentPart{type: :text, text: "Found it."}
             ] = response.message.content

      assert response.provider_meta["stop_reason"] == "pause_turn"
    end
  end

  describe "code-execution container round-trip" do
    @container %{"id" => "container_test123", "expires_at" => "2026-01-01T00:00:00Z"}

    test "non-streaming decode keeps the container in provider_meta" do
      data =
        server_tool_response_data("pause_turn")
        |> Map.put("container", @container)

      {:ok, response} = Anthropic.Response.decode_response(data, model())

      assert response.provider_meta["container"] == @container
    end

    test "streaming message_start surfaces the container as a meta chunk" do
      event = %{
        data: %{
          "type" => "message_start",
          "message" => %{"id" => "msg_test", "container" => @container, "usage" => %{}}
        }
      }

      {chunks, _state} =
        Anthropic.decode_stream_event(event, model(), Anthropic.init_stream_state(model()))

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: %{container: @container}}] = chunks
    end

    test "streaming message_delta surfaces a mid-turn container as a meta chunk" do
      event = %{
        data: %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => "pause_turn", "container" => @container},
          "usage" => %{}
        }
      }

      {chunks, _state} =
        Anthropic.decode_stream_event(event, model(), Anthropic.init_stream_state(model()))

      assert Enum.any?(chunks, &match?(%{metadata: %{container: @container}}, &1))
    end

    test "container meta chunks survive into the built response's provider_meta" do
      chunks = [
        ReqLLM.StreamChunk.meta(%{container: @container}),
        ReqLLM.StreamChunk.text("Working on it."),
        ReqLLM.StreamChunk.meta(%{
          finish_reason: :incomplete,
          stop_reason: "pause_turn",
          terminal?: true
        })
      ]

      {:ok, response} =
        ReqLLM.Provider.Defaults.ResponseBuilder.build_response(
          chunks,
          %{finish_reason: :incomplete},
          context: %ReqLLM.Context{messages: []},
          model: model()
        )

      assert response.provider_meta["container"] == @container
    end

    test "encode_body sends anthropic_container as the top-level container param" do
      context = %ReqLLM.Context{
        messages: [
          %ReqLLM.Message{role: :user, content: [ContentPart.text("resume")], metadata: %{}}
        ]
      }

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model().model,
          stream: false,
          anthropic_container: "container_test123"
        ]
      }

      updated_request = Anthropic.encode_body(mock_request)
      decoded = ReqLLM.Test.Helpers.json_body(updated_request)

      assert decoded["container"] == "container_test123"
    end

    test "encode_body omits container when the option is absent" do
      context = %ReqLLM.Context{
        messages: [
          %ReqLLM.Message{role: :user, content: [ContentPart.text("hi")], metadata: %{}}
        ]
      }

      mock_request = %Req.Request{
        options: [context: context, model: model().model, stream: false]
      }

      updated_request = Anthropic.encode_body(mock_request)
      decoded = ReqLLM.Test.Helpers.json_body(updated_request)

      refute Map.has_key?(decoded, "container")
    end
  end

  describe "client tool_use block completion tracking" do
    test "content_block_stop of a tool_use block emits a tool_call_complete marker" do
      events = [
        %{
          data: %{
            "type" => "content_block_start",
            "index" => 0,
            "content_block" => %{
              "type" => "tool_use",
              "id" => "toolu_1",
              "name" => "list_channels"
            }
          }
        },
        %{data: %{"type" => "content_block_stop", "index" => 0}}
      ]

      {chunks, _state} =
        Enum.reduce(events, {[], Anthropic.init_stream_state(model())}, fn event, {acc, state} ->
          {event_chunks, next_state} = Anthropic.decode_stream_event(event, model(), state)
          {acc ++ event_chunks, next_state}
        end)

      assert Enum.any?(chunks, &match?(%{metadata: %{tool_call_complete: 0}}, &1))
    end

    test "a tool_use block the stream never closes emits no completion marker" do
      event = %{
        data: %{
          "type" => "content_block_start",
          "index" => 0,
          "content_block" => %{"type" => "tool_use", "id" => "toolu_1", "name" => "list_channels"}
        }
      }

      {chunks, _state} =
        Anthropic.decode_stream_event(event, model(), Anthropic.init_stream_state(model()))

      refute Enum.any?(chunks, &match?(%{metadata: %{tool_call_complete: _}}, &1))
    end

    test "content_block_stop of a server block still emits the completed block, not a tool marker" do
      events = [
        %{
          data: %{
            "type" => "content_block_start",
            "index" => 0,
            "content_block" => %{
              "type" => "server_tool_use",
              "id" => "srvtoolu_1",
              "name" => "web_search",
              "input" => %{}
            }
          }
        },
        %{data: %{"type" => "content_block_stop", "index" => 0}}
      ]

      {chunks, _state} =
        Enum.reduce(events, {[], Anthropic.init_stream_state(model())}, fn event, {acc, state} ->
          {event_chunks, next_state} = Anthropic.decode_stream_event(event, model(), state)
          {acc ++ event_chunks, next_state}
        end)

      assert Enum.any?(
               chunks,
               &match?(%{metadata: %{provider_block: %{"type" => "server_tool_use"}}}, &1)
             )

      refute Enum.any?(chunks, &match?(%{metadata: %{tool_call_complete: _}}, &1))
    end
  end

  describe "code_execution server tool" do
    test "encode_body adds the code_execution tool" do
      context = %ReqLLM.Context{
        messages: [
          %ReqLLM.Message{role: :user, content: [ContentPart.text("run it")], metadata: %{}}
        ]
      }

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model().model,
          stream: false,
          provider_options: [code_execution: %{}]
        ]
      }

      updated_request = Anthropic.encode_body(mock_request)
      decoded = ReqLLM.Test.Helpers.json_body(updated_request)

      assert [%{"type" => "code_execution_20260521", "name" => "code_execution"}] =
               decoded["tools"]
    end
  end
end
