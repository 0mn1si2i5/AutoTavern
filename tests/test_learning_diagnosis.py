import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omnibob.learning_diagnosis import fit_normalization, future_terminal, loss_parts


def test_terminal_return_never_crosses_reset_or_invents_unfinished_reward():
    rewards = np.array([[-.1, -.1], [.9, -.1], [-.1, -.1], [-1.1, -.1]])
    dones = np.array([[False, False], [True, False], [False, False], [True, False]])
    truth, distance = future_terminal(rewards, dones, .1)
    np.testing.assert_allclose(truth[:, 0], [.8, .9, -1.2, -1.1])
    np.testing.assert_array_equal(distance[:, 0], [1, 0, 1, 0])
    assert np.isnan(truth[:, 1]).all()
    assert (distance[:, 1] == -1).all()


class TwoActions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.zeros(2))
        self.v = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, f, c, mask):
        return self.scores.expand(len(f), -1), self.v.expand(len(f))


def test_actor_reward_sign_and_reference_gradient_at_anchor():
    model = TwoActions()
    data = dict(features=torch.zeros(1, 1), cards=torch.zeros(1, 1), action_mask=torch.ones(1, 2),
                actions=torch.tensor([0]), logp=torch.tensor([-np.log(2)], dtype=torch.float32),
                advantages=torch.tensor([1.]), returns=torch.tensor([1.]),
                reference_logp=torch.full((1, 2), -np.log(2)))
    parts, _, _, kl = loss_parts(model, data, torch.tensor([0]))
    actor_grad = torch.autograd.grad(parts["actor"], model.scores, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(parts["reference"], model.scores, retain_graph=True)[0]
    value_grad = torch.autograd.grad(parts["critic"], model.v)[0]
    assert actor_grad[0] < 0 < actor_grad[1]  # Gradient descent raises the rewarded action's probability.
    assert value_grad < 0  # Gradient descent raises the underestimated value.
    torch.testing.assert_close(reference_grad, torch.zeros(2))
    assert kl == 0
    data["advantages"] = torch.tensor([-1.])
    negative, *_ = loss_parts(model, data, torch.tensor([0]))
    grad = torch.autograd.grad(negative["actor"], model.scores)[0]
    torch.testing.assert_close(grad, -actor_grad)


def test_held_rewards_cannot_change_fit_advantage_normalization():
    raw = torch.tensor([1., 2., 3., 4.])
    fit = torch.tensor([0, 1])
    before, *_ = fit_normalization(raw, fit)
    raw[2:] = torch.tensor([-1000., 1000.])
    after, *_ = fit_normalization(raw, fit)
    torch.testing.assert_close(after[fit], before[fit])
    torch.testing.assert_close(after[fit], torch.tensor([-1., 1.]))


def test_micro_state_removes_skipped_turn_hero_and_executes_correct_buy():
    from omnibob.learning_microdiagnosis import make_states
    # This exact seed used to choose Faelin and correctly reject all BUY actions.
    data, metadata = make_states(127000078, 1, 'BG26_OG_248')
    target = int(data['labels'][0])
    assert data['action_mask'][0, 8+target]
    assert metadata[0]['large'] > metadata[0]['small']
