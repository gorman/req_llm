defmodule ReqLLM.FinchRequestAdapter do
  @moduledoc """
  Behaviour for transforming `Finch.Request` structs just before a streaming
  request is sent.

  This is the config-level counterpart to the per-request `on_finch_request`
  option. Because config files cannot hold anonymous functions, the adapter
  must be a module that implements this behaviour.

  It is particularly useful for environment-specific concerns that should apply
  globally — for example injecting trace headers in a test environment — without
  touching individual call sites.

  ## Precedence

  Both mechanisms can be combined. The config-level adapter is applied first,
  then the per-request `on_finch_request` callback (if given). Each step
  receives the output of the previous one. If the adapter refuses the request
  with `{:error, reason}`, the callback does not run.

  ## Configuration

      # config/test.exs
      config :req_llm, finch_request_adapter: MyApp.TestFinchAdapter

  ## Example

      defmodule MyApp.TestFinchAdapter do
        @behaviour ReqLLM.FinchRequestAdapter

        @impl true
        def call(%Finch.Request{} = request) do
          %{request | headers: request.headers ++ [{"x-test-env", "true"}]}
        end
      end

  """

  @doc """
  Transform a `Finch.Request` just before it is sent.

  The request has already been fully built by the provider (authentication,
  body encoding, base headers). Return a `Finch.Request` — either the original
  or a modified copy.

  Return `{:error, reason}` to stop the request instead of sending it. The
  reason travels the same path as any other build failure, so the caller gets
  `{:error, {:provider_build_failed, reason}}` rather than a stream that dies
  mid-flight. This is the only seam that sees the assembled body, so it is the
  only place a size or content guard can act on what is actually going to be
  sent.
  """
  @callback call(Finch.Request.t()) :: Finch.Request.t() | {:error, term()}
end
