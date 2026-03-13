"""Unit tests for scripts.plot_agent_communication."""

from scripts.plot_agent_communication import _messaging_summary


class TestMessagingSummary:
    def test_formats_messaging_controls(self):
        summary = _messaging_summary(
            {
                "messaging_controls": {
                    "enabled": True,
                    "max_message_chars": 400,
                    "max_inbox_messages": 20,
                    "max_sends_per_agent_per_round": 3,
                    "max_broadcasts_per_round": 20,
                    "ttl_rounds": 10,
                }
            }
        )
        assert summary is not None
        assert "Messaging controls:" in summary
        assert "max_chars=400" in summary
        assert "ttl_rounds=10" in summary

    def test_returns_none_without_messaging_payload(self):
        assert _messaging_summary({}) is None
