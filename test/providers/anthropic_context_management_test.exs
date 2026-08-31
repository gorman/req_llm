defmodule ReqLLM.Providers.AnthropicContextManagementTest do
  @moduledoc """
  The `context_management` request parameter: it reaches the body untouched,
  each edit type brings its own beta header, and the API's report of what it
  removed survives a stream.
  """
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.Anthropic

  @api_key "anthropic-context-management-test-key"

  @clear %{
    edits: [
      %{
        type: "clear_tool_uses_20250919",
        trigger: %{type: "input_tokens", value: 600_000},
        keep: %{type: "tool_uses", value: 6}
      }
    ]
  }

  defp model do
    {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-6")
    model
  end

  defp body_for(opts) do
    {:ok, request} =
      Anthropic.prepare_request(
        :chat,
        model(),
        "Hello",
        [api_key: @api_key, max_tokens: 64] ++ opts
      )

    request |> Anthropic.encode_body() |> ReqLLM.Test.Helpers.json_body()
  end

  defp beta_header(opts) do
    {:ok, request} =
      Anthropic.prepare_request(
        :chat,
        model(),
        "Hello",
        [api_key: @api_key, max_tokens: 64] ++ opts
      )

    request
    |> Req.Request.get_header("anthropic-beta")
    |> List.first()
    |> to_string()
    |> String.split(",")
  end

  describe "request body" do
    test "omits context_management when it was not asked for" do
      refute Map.has_key?(body_for([]), "context_management")
    end

    test "passes the edits through untouched" do
      assert body_for(context_management: @clear)["context_management"] == %{
               "edits" => [
                 %{
                   "type" => "clear_tool_uses_20250919",
                   "trigger" => %{"type" => "input_tokens", "value" => 600_000},
                   "keep" => %{"type" => "tool_uses", "value" => 6}
                 }
               ]
             }
    end

    test "reads it from provider_options too" do
      assert body_for(provider_options: [context_management: @clear])["context_management"] ==
               body_for(context_management: @clear)["context_management"]
    end

    test "ignores an empty config rather than sending a bare object" do
      refute Map.has_key?(body_for(context_management: %{}), "context_management")
    end
  end

  describe "beta headers" do
    test "a clearing edit brings the context-management flag" do
      assert "context-management-2025-06-27" in beta_header(context_management: @clear)
    end

    test "a compaction edit brings its own flag" do
      edits = %{edits: [%{type: "compact_20260112"}]}

      assert "compact-2026-01-12" in beta_header(context_management: edits)
    end

    # The option's own validation requires atom keys on the outer map, but an
    # edit built from stored JSON arrives string-keyed. Missing its header
    # fails the whole request, so read both conventions.
    test "reads a string-keyed edit" do
      edits = %{edits: [%{"type" => "clear_tool_uses_20250919"}]}

      assert "context-management-2025-06-27" in beta_header(context_management: edits)
    end

    test "two edits sharing a flag name it once" do
      edits = %{
        edits: [%{type: "clear_tool_uses_20250919"}, %{type: "clear_thinking_20251015"}]
      }

      header = beta_header(context_management: edits)

      assert Enum.count(header, &(&1 == "context-management-2025-06-27")) == 1
    end

    test "no context_management, no flag" do
      refute "context-management-2025-06-27" in beta_header([])
    end
  end

  describe "streaming response" do
    test "the closing message_delta reports what was cleared" do
      event = %{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn"},
        "usage" => %{"output_tokens" => 5},
        "context_management" => %{
          "applied_edits" => [
            %{
              "type" => "clear_tool_uses_20250919",
              "cleared_tool_uses" => 8,
              "cleared_input_tokens" => 50_000
            }
          ]
        }
      }

      chunks = Anthropic.Response.decode_stream_event(%{data: event}, model())

      assert Enum.any?(chunks, fn chunk ->
               match?(
                 %{"applied_edits" => [%{"cleared_input_tokens" => 50_000} | _]},
                 chunk.metadata[:context_management]
               )
             end)
    end

    test "a message_delta without a report adds no chunk" do
      event = %{"type" => "message_delta", "delta" => %{"stop_reason" => "end_turn"}}

      chunks = Anthropic.Response.decode_stream_event(%{data: event}, model())

      refute Enum.any?(chunks, &Map.has_key?(&1.metadata, :context_management))
    end
  end
end
