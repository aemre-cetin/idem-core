import torch
import idempotent_core

def test_native_version():
    ver = idempotent_core.get_native_version()
    print(f"Native Engine Version: {ver}")
    assert "c20" in ver or "1.0.0" in ver

def test_cpu_float32_compaction():
    torch.manual_seed(42)
    N, D, K = 1024, 64, 256
    data = torch.randn((N, D), dtype=torch.float32)
    scores = torch.rand(N, dtype=torch.float32)

    target_map = idempotent_core.generate_idempotent_map(scores, K)

    # Reference gather
    expected = data[target_map[:K].to(torch.long)].clone()

    # In-place C++
    test_data = data.clone()
    idempotent_core.compact_inplace(test_data, target_map)
    actual = test_data[:K]

    diff = torch.max(torch.abs(expected - actual)).item()
    print(f"CPU Float32 Diff: {diff}")
    assert diff == 0.0

def test_cpu_float16_compaction():
    torch.manual_seed(42)
    N, D, K = 512, 128, 128
    data = torch.randn((N, D), dtype=torch.float16)
    scores = torch.rand(N, dtype=torch.float32)

    target_map = idempotent_core.generate_idempotent_map(scores, K)

    expected = data[target_map[:K].to(torch.long)].clone()
    test_data = data.clone()
    idempotent_core.compact_inplace(test_data, target_map)
    actual = test_data[:K]

    diff = torch.max(torch.abs(expected - actual)).item()
    print(f"CPU Float16 Diff: {diff}")
    assert diff == 0.0

if __name__ == "__main__":
    test_native_version()
    test_cpu_float32_compaction()
    test_cpu_float16_compaction()
    print("ALL NATIVE BINDING TESTS PASSED!")