defmodule ReqLLM.Providers.GoogleFinchTest do
  use ExUnit.Case, async: true

  @nested_finch_options Version.match?(to_string(Application.spec(:req, :vsn)), ">= 0.7.0")

  alias ReqLLM.Providers.Google

  describe "attach/3 Finch pool routing" do
    test "routes the request to the configured Finch pool" do
      request = Google.attach(Req.new(), "google:gemini-2.5-flash", api_key: "test")

      if @nested_finch_options do
        assert request.options[:finch] == [name: ReqLLM.Application.finch_name()]
      else
        assert request.options[:finch] == ReqLLM.Application.finch_name()
      end
    end

    test "keeps a Finch pool already set on the request" do
      request =
        Req.new()
        |> Req.Request.merge_options(finch: MyApp.CustomFinch)
        |> Google.attach("google:gemini-2.5-flash", api_key: "test")

      if @nested_finch_options do
        assert request.options[:finch] == [name: MyApp.CustomFinch]
      else
        assert request.options[:finch] == MyApp.CustomFinch
      end
    end
  end
end
