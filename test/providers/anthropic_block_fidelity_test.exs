defmodule ReqLLM.Providers.AnthropicBlockFidelityTest do
  @moduledoc """
  An assistant turn with signed thinking is replayed with exactly the content
  array Anthropic returned.

  Newer models sign each thinking block over everything before it: the system
  prompt, the tools, earlier messages, and the blocks that came before it in the
  same assistant message. When the replay moves, merges, or drops a block, the
  API rejects the next request ("Content that preceded this block ... is
  missing"). This turn is the order seen from claude-sonnet-5-5: text before the
  search, thinking after the result, and two text blocks (one with citations)
  before the tool call.
  """
  use ExUnit.Case, async: true

  alias ReqLLM.Message
  alias ReqLLM.Provider.ResponseBuilder
  alias ReqLLM.Providers.Anthropic

  @citation %{
    "type" => "web_search_result_location",
    "url" => "https://example.com/forecast",
    "title" => "Forecast",
    "encrypted_index" => "IDX-1",
    "cited_text" => "Highs in the low 80s."
  }

  @blocks [
    %{"type" => "text", "text" => "Let me look that up."},
    %{
      "type" => "server_tool_use",
      "id" => "srvtoolu_1",
      "name" => "web_search",
      "input" => %{"query" => "portland weather"}
    },
    %{
      "type" => "web_search_tool_result",
      "tool_use_id" => "srvtoolu_1",
      "content" => [
        %{
          "type" => "web_search_result",
          "title" => "Forecast",
          "url" => "https://example.com/forecast",
          "encrypted_content" => "OPAQUE"
        }
      ]
    },
    %{"type" => "thinking", "thinking" => "Highs are in the low 80s.", "signature" => "SIG-A"},
    %{"type" => "redacted_thinking", "data" => "REDACTED-DATA"},
    %{"type" => "text", "text" => "Here is the forecast. "},
    %{"type" => "text", "text" => "Highs in the low 80s.", "citations" => [@citation]},
    %{
      "type" => "tool_use",
      "id" => "toolu_1",
      "name" => "save_note",
      "input" => %{"note" => "low 80s"}
    }
  ]

  defp model do
    {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-6")
    model
  end

  defp sse_events(blocks) do
    block_events =
      blocks
      |> Enum.with_index()
      |> Enum.flat_map(fn {block, index} -> block_events(block, index) end)

    [
      event(%{
        "type" => "message_start",
        "message" => %{"id" => "msg_1", "usage" => %{"input_tokens" => 10}}
      })
    ] ++
      block_events ++
      [
        event(%{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => "tool_use"},
          "usage" => %{"output_tokens" => 20}
        }),
        event(%{"type" => "message_stop"})
      ]
  end

  defp block_events(%{"type" => "text", "text" => text} = block, index) do
    {head, tail} = String.split_at(text, div(String.length(text), 2))

    citation_deltas =
      for citation <- Map.get(block, "citations", []) do
        delta(index, %{"type" => "citations_delta", "citation" => citation})
      end

    [start(index, %{"type" => "text", "text" => ""})] ++
      citation_deltas ++
      [
        delta(index, %{"type" => "text_delta", "text" => head}),
        delta(index, %{"type" => "text_delta", "text" => tail}),
        stop(index)
      ]
  end

  defp block_events(%{"type" => "thinking"} = block, index) do
    [
      start(index, %{"type" => "thinking", "thinking" => ""}),
      delta(index, %{"type" => "thinking_delta", "thinking" => block["thinking"]}),
      delta(index, %{"type" => "signature_delta", "signature" => block["signature"]}),
      stop(index)
    ]
  end

  defp block_events(%{"type" => type} = block, index)
       when type in ["server_tool_use", "tool_use"] do
    [
      start(index, Map.put(block, "input", %{})),
      delta(index, %{
        "type" => "input_json_delta",
        "partial_json" => Jason.encode!(block["input"])
      }),
      stop(index)
    ]
  end

  defp block_events(block, index), do: [start(index, block), stop(index)]

  defp start(index, block),
    do: event(%{"type" => "content_block_start", "index" => index, "content_block" => block})

  defp delta(index, delta),
    do: event(%{"type" => "content_block_delta", "index" => index, "delta" => delta})

  defp stop(index), do: event(%{"type" => "content_block_stop", "index" => index})

  defp event(data), do: %{data: data}

  defp stream_message(blocks) do
    {chunks, state} =
      Enum.reduce(sse_events(blocks), {[], Anthropic.init_stream_state(model())}, fn event,
                                                                                     {acc, state} ->
        {event_chunks, next_state} = Anthropic.decode_stream_event(event, model(), state)
        {acc ++ event_chunks, next_state}
      end)

    {flushed, _state} = Anthropic.flush_stream_state(model(), state)

    {:ok, response} =
      ResponseBuilder.for_model(model()).build_response(
        chunks ++ flushed,
        %{finish_reason: :tool_calls},
        context: %ReqLLM.Context{messages: []},
        model: model()
      )

    response.message
  end

  defp buffered_message(blocks) do
    body = %{
      "id" => "msg_1",
      "type" => "message",
      "role" => "assistant",
      "model" => "claude-sonnet-4-6",
      "content" => blocks,
      "stop_reason" => "tool_use",
      "usage" => %{"input_tokens" => 10, "output_tokens" => 20}
    }

    {:ok, response} = Anthropic.Response.decode_response(body, model())
    response.message
  end

  # What jido_ai replays: only content, tool_calls and reasoning_details survive
  # its projection into the next request, not the message metadata.
  defp as_replayed(%Message{} = message) do
    %Message{
      role: :assistant,
      content: message.content,
      tool_calls: message.tool_calls,
      reasoning_details: message.reasoning_details
    }
  end

  defp wire_assistant_content(message) do
    context = %ReqLLM.Context{
      messages: [
        ReqLLM.Context.user("What is the weather in Portland?"),
        message,
        ReqLLM.Context.tool_result("toolu_1", "saved")
      ]
    }

    %{messages: [_user, %{role: "assistant", content: content}, _tool]} =
      Anthropic.Context.encode_request(context, model())

    content |> Jason.encode!() |> Jason.decode!()
  end

  describe "streaming" do
    test "the replayed content is the original block array" do
      message = stream_message(@blocks)

      assert wire_assistant_content(message) == @blocks
      assert wire_assistant_content(as_replayed(message)) == @blocks
    end

    test "the response text is still the joined text blocks" do
      response_text =
        stream_message(@blocks).content
        |> Enum.filter(&(&1.type == :text))
        |> Enum.map_join(& &1.text)

      assert response_text == "Let me look that up.Here is the forecast. Highs in the low 80s."
    end

    test "a block type the decoder does not know is kept in place" do
      blocks = List.insert_at(@blocks, 4, %{"type" => "future_block", "payload" => "opaque"})

      assert wire_assistant_content(stream_message(blocks)) == blocks
    end
  end

  describe "non-streaming" do
    test "the replayed content is the original block array" do
      message = buffered_message(@blocks)

      assert wire_assistant_content(message) == @blocks
      assert wire_assistant_content(as_replayed(message)) == @blocks
    end

    test "a block type the decoder does not know is kept in place" do
      blocks = List.insert_at(@blocks, 4, %{"type" => "future_block", "payload" => "opaque"})

      assert wire_assistant_content(buffered_message(blocks)) == blocks
    end
  end
end
