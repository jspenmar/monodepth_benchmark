import torch.utils.data._utils.collate

from src.utils import default_collate, MultiLevelTimer


class TestDefaultCollate:
    def test_base(self):
        torch_collate = torch.utils.data._utils.collate.default_collate

        input = [torch.rand(3, 100, 200) for _ in range(5)]
        target = torch_collate(input)
        out = default_collate(input)
        assert out.allclose(target), "Error when matching default tensor collate."

        input = ['test' for _ in range(5)]
        target = torch_collate(input)
        out = default_collate(input)
        assert input == target == out, "Error when matching default string collate."

    def test_timer(self):
        input = [MultiLevelTimer() for _ in range(5)]
        out = default_collate(input)
        assert input == out, "Error when matching MultiLevelTimer collate."
