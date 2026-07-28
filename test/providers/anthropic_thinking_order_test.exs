defmodule ReqLLM.Providers.AnthropicThinkingOrderTest do
  @moduledoc """
  Thinking blocks keep their original position relative to server-tool blocks
  across a decode → context → encode round-trip.

  Anthropic thinks again after each server-tool result, so a web_search turn with
  thinking enabled comes back as:

      [thinking, server_tool_use, web_search_tool_result, thinking, text, tool_use]

  Those blocks must be replayed exactly as received. Bucketing them by type
  (`thinking ++ text ++ tool`) hoists the second thinking block from index 3 to
  index 1, and Anthropic rejects the next request with "`thinking` or
  `redacted_thinking` blocks in the latest assistant message cannot be modified",
  naming `messages.1.content.1` — the hoisted block's new index.
  """
  use ExUnit.Case, async: true

  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Providers.Anthropic

  defp model do
    {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-6")
    model
  end

  # The interleaved shape, as SSE events.
  defp interleaved_events do
    [
      # 0: thinking
      %{
        data: %{
          "type" => "content_block_start",
          "index" => 0,
          "content_block" => %{"type" => "thinking", "thinking" => ""}
        }
      },
      %{
        data: %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{"type" => "thinking_delta", "thinking" => "I should search."}
        }
      },
      %{
        data: %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{"type" => "signature_delta", "signature" => "SIG-A"}
        }
      },
      %{data: %{"type" => "content_block_stop", "index" => 0}},
      # 1: server_tool_use
      %{
        data: %{
          "type" => "content_block_start",
          "index" => 1,
          "content_block" => %{
            "type" => "server_tool_use",
            "id" => "srvtoolu_1",
            "name" => "web_search",
            "input" => %{"query" => "portland weather"}
          }
        }
      },
      %{data: %{"type" => "content_block_stop", "index" => 1}},
      # 2: web_search_tool_result
      %{
        data: %{
          "type" => "content_block_start",
          "index" => 2,
          "content_block" => %{
            "type" => "web_search_tool_result",
            "tool_use_id" => "srvtoolu_1",
            "content" => [
              %{
                "type" => "web_search_result",
                "title" => "Forecast",
                "url" => "https://example.com",
                "encrypted_content" => "OPAQUE"
              }
            ]
          }
        }
      },
      %{data: %{"type" => "content_block_stop", "index" => 2}},
      # 3: thinking AGAIN — this is the block that gets hoisted
      %{
        data: %{
          "type" => "content_block_start",
          "index" => 3,
          "content_block" => %{"type" => "thinking", "thinking" => ""}
        }
      },
      %{
        data: %{
          "type" => "content_block_delta",
          "index" => 3,
          "delta" => %{"type" => "thinking_delta", "thinking" => "Highs low 80s."}
        }
      },
      %{
        data: %{
          "type" => "content_block_delta",
          "index" => 3,
          "delta" => %{"type" => "signature_delta", "signature" => "SIG-B"}
        }
      },
      %{data: %{"type" => "content_block_stop", "index" => 3}},
      # 4: text
      %{
        data: %{
          "type" => "content_block_start",
          "index" => 4,
          "content_block" => %{"type" => "text", "text" => ""}
        }
      },
      %{
        data: %{
          "type" => "content_block_delta",
          "index" => 4,
          "delta" => %{"type" => "text_delta", "text" => "Here are the blocks:"}
        }
      },
      %{data: %{"type" => "content_block_stop", "index" => 4}}
    ]
  end

  defp decode_stream(events) do
    {chunks, _state} =
      Enum.reduce(events, {[], Anthropic.init_stream_state(model())}, fn event, {acc, state} ->
        {event_chunks, next_state} = Anthropic.decode_stream_event(event, model(), state)
        {acc ++ event_chunks, next_state}
      end)

    chunks
  end

  defp materialize(chunks) do
    {:ok, response} =
      ReqLLM.Provider.Defaults.ResponseBuilder.build_response(
        chunks ++
          [ReqLLM.StreamChunk.meta(%{finish_reason: :stop, terminal?: true})],
        %{finish_reason: :stop},
        context: %ReqLLM.Context{messages: []},
        model: model()
      )

    response.message
  end

  defp encoded_block_types(message) do
    %{messages: [%{content: blocks}]} =
      Anthropic.Context.encode_request(%ReqLLM.Context{messages: [message]}, model())

    Enum.map(blocks, fn
      %{"type" => t} -> t
      %{type: t} -> t
    end)
  end

  describe "streaming decode" do
    test "thinking blocks land positionally, interleaved with the server-tool blocks" do
      content = decode_stream(interleaved_events()) |> materialize() |> Map.fetch!(:content)

      # `thinking_delta` also emits `:thinking` StreamChunks so callers can stream
      # reasoning to a UI, and the builder appends those as one trailing
      # `ContentPart{type: :thinking}`. It is inert for Anthropic: the encoder has
      # no `:thinking` clause, so it is dropped — asserted below in
      # "the inert :thinking part is not sent to Anthropic". Only the positional
      # provider blocks matter here.
      positional =
        Enum.flat_map(content, fn
          # Thinking blocks are atom-keyed, server-tool blocks string-keyed.
          %ContentPart{type: :provider_block, data: %{type: t}} -> [t]
          %ContentPart{type: :provider_block, data: %{"type" => t}} -> [t]
          %ContentPart{type: :text, text: ""} -> []
          %ContentPart{type: :text} -> ["text"]
          _ -> []
        end)

      assert positional == [
               "thinking",
               "server_tool_use",
               "web_search_tool_result",
               "thinking",
               "text"
             ]
    end

    test "the inert :thinking part is not sent to Anthropic" do
      message = decode_stream(interleaved_events()) |> materialize()

      assert Enum.any?(message.content, &match?(%ContentPart{type: :thinking}, &1)),
             "expected the live-display thinking part to exist"

      # Exactly the two signed provider blocks reach the wire, not three.
      assert Enum.count(encoded_block_types(message), &(&1 == "thinking")) == 2
    end

    test "the signature rides along on each thinking block" do
      content = decode_stream(interleaved_events()) |> materialize() |> Map.fetch!(:content)

      sigs =
        for %ContentPart{type: :provider_block, data: %{type: "thinking"} = b} <- content,
            do: b[:signature]

      assert sigs == ["SIG-A", "SIG-B"]
    end

    test "reasoning_details is still populated for downstream consumers" do
      message = decode_stream(interleaved_events()) |> materialize()

      assert length(message.reasoning_details) == 2
      assert Enum.map(message.reasoning_details, & &1.signature) == ["SIG-A", "SIG-B"]
    end
  end

  describe "re-encode into the next request" do
    test "order is preserved — this is the 400 that broke every web_search turn" do
      types = decode_stream(interleaved_events()) |> materialize() |> encoded_block_types()

      assert types == [
               "thinking",
               "server_tool_use",
               "web_search_tool_result",
               "thinking",
               "text"
             ]
    end

    test "thinking is not duplicated by the reasoning_details prepend" do
      types = decode_stream(interleaved_events()) |> materialize() |> encoded_block_types()

      assert Enum.count(types, &(&1 == "thinking")) == 2
    end

    test "signatures survive the round-trip" do
      message = decode_stream(interleaved_events()) |> materialize()

      %{messages: [%{content: blocks}]} =
        Anthropic.Context.encode_request(%ReqLLM.Context{messages: [message]}, model())

      sigs = for %{type: "thinking"} = b <- blocks, do: b[:signature]
      assert sigs == ["SIG-A", "SIG-B"]
    end
  end

  describe "regression guards" do
    test "a turn with no server tools still encodes thinking first" do
      events =
        Enum.take(interleaved_events(), 4) ++
          [
            %{
              data: %{
                "type" => "content_block_start",
                "index" => 1,
                "content_block" => %{"type" => "text", "text" => ""}
              }
            },
            %{
              data: %{
                "type" => "content_block_delta",
                "index" => 1,
                "delta" => %{"type" => "text_delta", "text" => "No search needed."}
              }
            },
            %{data: %{"type" => "content_block_stop", "index" => 1}}
          ]

      types = decode_stream(events) |> materialize() |> encoded_block_types()
      assert types == ["thinking", "text"]
    end

    test "a turn with no thinking blocks at all is unaffected" do
      events = Enum.drop(interleaved_events(), 4) |> Enum.reject(&(&1.data["index"] == 3))

      types = decode_stream(events) |> materialize() |> encoded_block_types()
      assert types == ["server_tool_use", "web_search_tool_result", "text"]
    end
  end
end
