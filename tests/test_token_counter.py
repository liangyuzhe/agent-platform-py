from unittest.mock import patch


def test_token_counter_falls_back_when_tiktoken_encoding_unavailable():
    from agents.tool.token_counter import TokenCounter

    with patch("agents.tool.token_counter.tiktoken.get_encoding", side_effect=OSError("offline")):
        counter = TokenCounter()

    assert counter.count("hello world") == 2
    assert counter.count("去年亏损") == 4
