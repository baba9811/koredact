"""Compatibility, accelerator fallback and ordered worker batches on the public tiny bundle."""
from pathlib import Path
import koredact

bundle=Path(__file__).resolve().parents[1]/'fixtures/tiny_bundle'
texts=['문의 010-1234-5678 홍길동', '', 'a@b.test #{이름}', '문의 12345678 '*45]
cpu=koredact.Masker.from_pretrained(str(bundle),device='cpu',threads=1,mask_tokens={'NUM':'***'})
assert cpu.device=='cpu'
expected=[cpu.mask(text,backstop=True) for text in texts]
assert cpu.mask_many(texts,workers=2,backstop=True)==expected
assert cpu.predict_many(texts,workers=2,types=['NUM'],backstop=True)==[cpu.predict(text,types=['NUM'],backstop=True) for text in texts]
assert cpu.mask_many([],workers=2)==[]
assert cpu.mask_many(texts,workers=99,backstop=True)==expected
legacy=koredact.Masker.from_pretrained(str(bundle))
assert legacy.device in {'cpu','cuda','rocm','directml','openvino_gpu','qnn','openvino_npu','coreml','coreml_gpu','coreml_npu','directml_npu'}
for device in ('gpu','npu','metal','mps','coreml','cuda'):
    masker=koredact.Masker.from_pretrained(str(bundle),device=device)
    assert masker.mask(texts[0])==koredact.Masker.from_pretrained(str(bundle),device='cpu').mask(texts[0])
for kwargs in ({'device':'invalid'},{'threads':0}):
    try:koredact.Masker.from_pretrained(str(bundle),**kwargs)
    except ValueError:pass
    else:raise AssertionError('invalid options accepted')
try:cpu.mask_many(texts,workers=0)
except ValueError:pass
else:raise AssertionError('zero workers accepted')
try:cpu.mask_many(texts,workers=2,types=['INVALID'])
except ValueError:pass
else:raise AssertionError('invalid types accepted in batch')
print('runtime options and ordered worker checks passed')
