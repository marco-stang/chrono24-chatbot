from app.guards import TokenBudget


def test_budget_starts_full(tmp_path):
    budget = TokenBudget(tmp_path / "b.sqlite3", daily_limit=1000)
    assert budget.remaining() == 1000
    assert budget.used_today() == 0


def test_spend_reduces_remaining(tmp_path):
    budget = TokenBudget(tmp_path / "b.sqlite3", daily_limit=1000)
    budget.spend(300)
    budget.spend(200)
    assert budget.used_today() == 500
    assert budget.remaining() == 500


def test_budget_persists_across_instances(tmp_path):
    path = tmp_path / "b.sqlite3"
    TokenBudget(path, daily_limit=1000).spend(400)
    assert TokenBudget(path, daily_limit=1000).remaining() == 600


def test_remaining_can_go_negative(tmp_path):
    budget = TokenBudget(tmp_path / "b.sqlite3", daily_limit=100)
    budget.spend(250)
    assert budget.remaining() == -150
