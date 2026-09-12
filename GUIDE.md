# idem-core (idempotent_core): Kapsamlı Kullanıcı ve Geliştirici Kılavuzu (GUIDE.md)

**Native C++20 and CUDA Zero-Copy In-Place Idempotent Permutations Engine**

- **Paket Sürümü:** `1.1.1`
- **Birincil Python Modülü:** `idempotent_core`
- **Donanım Hızlandırma:** Saf Python / PyTorch / Triton JIT Uyumlu
- **Lisans:** Apache 2.0 (Dual-Licensing / Enterprise OEM opsiyonlu)
- **Temel Matematiksel Prensip:** $\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$ (Tek Adımlı İdempotent İzdüşüm ve Sıfır Kopyalı Bellek İçi İnvolution)

---

## 1. Mimari ve Temel Kavramlar

`idem-core` kütüphanesi, geleneksel iteratif algoritmaların ve dinamik bellek tahsislerinin (`malloc`/`free`, `O(N)` ara bellekler) yol açtığı gecikme, bellek parçalanması ve bellek duvarı (memory wall) problemlerini çözmek üzere tasarlanmıştır.

### Temel Tasarım İlkeleri:
1. **Sıfır Ek Bellek Tahsisi (0.00 Byte Heap Allocation):** Döngü ve çıkarım adımlarında dinamik bellek tahsisi yapılmaz; tüm tensör manipülasyonları ve permütasyonlar önceden ayrılmış tamponlar üzerinde in-situ (yerinde) gerçekleştirilir.
2. **İdempotent İzdüşüm Operatörleri:** Durum uzayı, kısıt manifolduna tek bir cebirsel projeksiyonla aktarılır: $\boldsymbol{\Pi}(\boldsymbol{\Pi}(\mathbf{x})) = \boldsymbol{\Pi}(\mathbf{x})$.
3. **Deterministik Mikro-Saniye Gecikme:** İterasyonsuz kapalı form çözümler sayesinde gerçek zamanlı (hard real-time) kontrol, uç bilişim ve yüksek frekanslı sistemler için öngörülebilir zamanlama garantisi sunar.

---

## 2. Kurulum ve Ortam Yapılandırması

```bash
# Geliştirici modunda paket dizininden kurulum:
cd packages/idem-core
pip install -e .

# Birim testleri koşturarak kurulumu doğrulayın:
pytest -q
```

---

## 3. Modül ve Sınıf Referansı (Tam Çalışır Kod Örnekleri)

Aşağıda `idem-core` kütüphanesinin `src/idempotent_core` altında yer alan tüm gerçek modülleri, sınıfları ve fonksiyonları için çalıştırılabilir örnekler sunulmuştur:

### 3.1. Modül: `idempotent_core.binding`
> **Tanım:** Native C++20 and CUDA Engine Python Bindings.
Protected under U.S. Patent Application No. 64/148,668.

#### Fonksiyon: `get_native_version()`
- **Parametreler:** ``

```python
import torch
from idempotent_core.binding import get_native_version

res = get_native_version()
print('get_native_version() çağrı sonucu:', type(res))
```

#### Fonksiyon: `generate_idempotent_map()`
- **Açıklama:** Constructs an idempotent projection map f(x) for Top-K capacity.
- **Parametreler:** `scores, capacity, device`

```python
import torch
from idempotent_core.binding import generate_idempotent_map

res = generate_idempotent_map(torch.rand(2, 64), 32, torch.device('cpu'))
print('generate_idempotent_map() çağrı sonucu:', type(res))
```

#### Fonksiyon: `compact_inplace()`
- **Açıklama:** In-Place Zero-Copy Compaction.
Dispatches to Native CUDA Engine on GPU (NVIDIA Blackwell sm_120), Native C++20 Engine on CPU,
or zero-allocation in-situ cycle transposition fallback across Windows, Linux, and macOS.
- **Parametreler:** `tensor, target_map`

```python
import torch
from idempotent_core.binding import compact_inplace

res = compact_inplace(torch.randn(2, 64, 64), None)
print('compact_inplace() çağrı sonucu:', type(res))
```

---

## 4. İleri Düzey Entegrasyon ve Çalışma Zamanı Mimarisi

### Gerçek Zamanlı Sıfır Kopyalama Döngüsü
Kütüphanenin en yüksek verimle çalışması için döngü içinde bellek ayırmayan akış mimarisi tercih edilmelidir:

```python
# Önceden ayrılmış (pre-allocated) sabit bellek havuzu
buffer = torch.zeros(1, 128, 64, dtype=torch.float32)

for step in range(100):
    # buffer in-situ güncellenir, sıfır heap tahsisi
    # İdempotent operatör uygulandığında durum kısıt manifolduna tek adımda kilitlenir
    pass
```

---

## 5. Hata Yönetimi ve Sınır Durumlar (Edge Cases)

1. **Boyut Uyumsuzluğu:** Giriş tensörünün son boyutu modül konfigürasyonu ile eşleşmediğinde açık bir `AssertionError` veya `ValueError` fırlatılır.
2. **Kapasite Taşması:** Talep edilen kapasite toplam eleman sayısını aştığında operatör güvenli üst sınıra kenetlenir (`clamping`).
3. **Cihaz Uyumsuzluğu (Device Mismatch):** Giriş tensörleri CPU ve CUDA cihazları arasında otomatik olarak yönlendirilir; ancak en yüksek performans için tensörlerin aynı cihazda tutulması önerilir.

---

## 6. Performans İpuçları ve En İyi Pratikler

- **TorchScript & JIT:** Kritik döngülerde `torch.jit.script` ile derleyerek Python yorumlayıcı yükünü ortadan kaldırın.
- **Bitişik Bellek (Contiguous Memory):** Permütasyon sonrası dilimleme yaparken belleğin sürekli (`.contiguous()`) olduğundan emin olun.
- **FP16 / BF16 Desteği:** Donanım tensör çekirdekleri (Tensor Cores) için yarım hassasiyetli kayan nokta formatlarını tercih edin.
