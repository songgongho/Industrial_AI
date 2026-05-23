# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors:** Dosovitskiy et al.  
**Year:** 2021  
**Venue:** ICLR

## What it does

이미지를 transformer 방식으로 표현합니다.

## Technique

Vision Transformer

## Strength

이미지 패치 단위로 결함을 넓게 볼 수 있습니다.

## Weakness

라벨이 적으면 학습이 불안정할 수 있습니다.

## Our Response

DINOv2 같은 사전학습 표현을 먼저 쓰고, 필요할 때만 미세조정합니다.

---

*Auto-generated from `src/research/guide.py`. Edit with care.*
