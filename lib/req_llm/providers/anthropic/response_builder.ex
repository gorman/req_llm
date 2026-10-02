defmodule ReqLLM.Providers.Anthropic.ResponseBuilder do
  @moduledoc """
  Anthropic-specific ResponseBuilder implementation.

  Handles Anthropic's specific requirements:
  - Content blocks must be non-empty when tool_calls are present
  - Maps `tool_use` finish reason to `:tool_calls`

  This fixes bug #269 where streaming tool-call-only responses
  produced empty content blocks that Anthropic's API rejected.
  """

  @behaviour ReqLLM.Provider.ResponseBuilder

  alias ReqLLM.Context
  alias ReqLLM.Message
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Provider.Defaults.ResponseBuilder, as: DefaultBuilder
  alias ReqLLM.StreamChunk

  @impl true
  def build_response(chunks, metadata, opts) do
    with {:ok, response} <- DefaultBuilder.build_response(chunks, metadata, opts) do
      {:ok, response |> restore_block_order(chunks) |> ensure_non_empty_content()}
    end
  end

  @doc false
  @spec build_buffered_response([ReqLLM.StreamChunk.t()], map(), keyword()) ::
          {:ok, ReqLLM.Response.t()} | {:error, term()}
  def build_buffered_response(chunks, metadata, opts) do
    with {:ok, response} <- DefaultBuilder.build_buffered_response(chunks, metadata, opts) do
      {:ok, response |> restore_block_order(chunks) |> preserve_empty_message(chunks, opts)}
    end
  end

  # A thinking signature covers every block before it in the same message, so
  # a turn with signed thinking must be replayed in the order Anthropic sent it.
  # The default builder puts provider blocks first and joins all text into one
  # part. The decoder marks each completed block in arrival order; rebuild the
  # replayable parts from those marks. Parts that are not replayed (the joined
  # :thinking text that Response.thinking/1 reads, or a parsed :object) stay
  # after them. Tool calls still come from `tool_calls`, which the encoder puts
  # last, where Anthropic sends them.
  #
  # When the marks do not hold all of the streamed text (a text block that never
  # closed), the default content is kept so that no text is lost.
  defp restore_block_order(%{message: %Message{} = message} = response, chunks) do
    ordered = Enum.flat_map(chunks, &ordered_part/1)

    if Enum.any?(ordered, &signed_thinking_part?/1) and
         joined_text(ordered) == streamed_text(chunks) do
      rest = Enum.reject(message.content, &replayable_part?/1)
      put_message(response, %{message | content: ordered ++ rest})
    else
      response
    end
  end

  defp restore_block_order(response, _chunks), do: response

  defp ordered_part(%StreamChunk{
         type: :meta,
         metadata: %{provider_block: block, provider: :anthropic}
       })
       when is_map(block),
       do: [ContentPart.provider_block(block, :anthropic)]

  defp ordered_part(%StreamChunk{
         type: :meta,
         metadata: %{anthropic_text_part: %ContentPart{} = part}
       }),
       do: [part]

  defp ordered_part(_chunk), do: []

  defp signed_thinking_part?(%ContentPart{type: :provider_block, data: block}) do
    case Map.get(block, :type) || Map.get(block, "type") do
      "thinking" -> is_binary(Map.get(block, :signature) || Map.get(block, "signature"))
      "redacted_thinking" -> true
      _type -> false
    end
  end

  defp signed_thinking_part?(_part), do: false

  defp joined_text(parts) do
    for %ContentPart{type: :text, text: text} <- parts, into: "", do: text
  end

  defp streamed_text(chunks) do
    for %StreamChunk{type: :content, text: text} when is_binary(text) <- chunks,
        into: "",
        do: text
  end

  defp replayable_part?(%ContentPart{type: type}), do: type in [:text, :provider_block]
  defp replayable_part?(_part), do: false

  defp put_message(
         %{message: old, context: %Context{messages: messages} = context} = response,
         new
       ) do
    messages =
      if List.last(messages) == old,
        do: List.replace_at(messages, -1, new),
        else: messages

    %{response | message: new, context: %{context | messages: messages}}
  end

  defp put_message(response, new), do: %{response | message: new}

  defp ensure_non_empty_content(%{message: %{tool_calls: tc, content: []}} = response)
       when is_list(tc) and tc != [] do
    content = [ContentPart.text("")]
    put_in(response.message.content, content)
  end

  defp ensure_non_empty_content(response), do: response

  defp preserve_empty_message(response, chunks, opts) do
    if Enum.any?(chunks, &message_chunk?/1) do
      response
    else
      %{response | message: nil, context: Keyword.fetch!(opts, :context)}
    end
  end

  defp message_chunk?(%ReqLLM.StreamChunk{type: type})
       when type in [:content, :thinking, :tool_call],
       do: true

  defp message_chunk?(_chunk), do: false
end
