import warnings
import torch

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model
from separability.data_classes import PruningConfig
from separability.prune import prune_and_evaluate

class TestPruneAndEvaluate:
    pruning_config = PruningConfig("nickypro/tinyllama-15m",
        attn_mode="pre-out", do_attn_mean_offset=False, use_accelerator=False,
        ff_frac=0.1, ff_eps=0.1, attn_frac=0.001, attn_eps=1e4,
        token_limit=1000, focus="pile", cripple="code")

    def __run_testing(self, _pruning_config: PruningConfig):
        c = _pruning_config
        opt = Model(c.model_repo, limit=c.token_limit, dtype="fp32",
                    use_accelerator=c.use_accelerator)
        data = prune_and_evaluate(opt, c)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss > 1
        assert code_loss > 1

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_and_evaluate(self, model_repo):
        c = self.pruning_config
        c.model_repo          = model_repo
        c.attn_mode           = "pre-out"
        c.use_accelerator     = False
        c.do_attn_mean_offset = False

        self.__run_testing(c)


    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_and_evaluate_accelerate(self, model_repo):
        print( "# Running Test: test_prune_and_evaluate_accelerate" )
        c = self.pruning_config
        c.model_repo          = model_repo
        c.attn_mode           = "pre-out"
        c.use_accelerator     = True
        c.do_attn_mean_offset = False

        if torch.cuda.device_count() <= 1:
            warnings.warn( "Multi-gpu not available", category=UserWarning )
            return

        self.__run_testing(c)

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_and_evaluate_mean_offset(self, model_repo):
        return
        # TODO: Fix mean offset
        c = self.pruning_config
        c.model_repo          = model_repo
        c.attn_mode           = "pre-out"
        c.use_accelerator     = False
        c.do_attn_mean_offset = True

        self.__run_testing(c)

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_attn_value_and_evaluate(self, model_repo):
        c = self.pruning_config
        c.model_repo          = model_repo
        c.attn_mode           = "value"
        c.use_accelerator     = False
        c.do_attn_mean_offset = False

        self.__run_testing(c)
