# Third-party attribution

Vision Scholar integrates [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness)
at commit `9b2efd795c6aa09f88b0c257d269a9e518da6ae7`, package version 0.1.9.
OpenHarness supplies the V1/V2 model/tool execution loop, shared message/event
DTOs, API clients, registry and permission checker. V3's `ResearchEngine` has
an independently implemented execution loop, context assembler and scheduler;
it continues to use OpenHarness interfaces and transport. The dependency is
pinned to this Git commit in `pyproject.toml` and `uv.lock`; no adjacent checkout
is needed. See `CONTRIBUTIONS.md` for the exact ownership boundary.

Other dependencies retain their own licenses; versions are recorded in
`uv.lock` and `frontend/package-lock.json`.

The ResNet, ViT, CLIP and DETR PDFs are downloaded from the authors' arXiv
pages for local reading. Each stored paper records its source URL and SHA256.
They are not included in this project's source repository and are not
relicensed by this notice. The digits experiment uses scikit-learn's bundled
UCI optical handwritten digits dataset; see `sklearn.datasets.load_digits`
for dataset provenance (UCI Optical Recognition of Handwritten Digits,
Alpaydin and Kaynak). Public evidence contains derived predictions and the
paired corrupted test inputs required for verification, not redistributed
paper PDFs or downloaded model weights. The dataset and papers retain their
original rights; the project MIT license does not relicense them.

## OpenHarness license

MIT License

Copyright (c) 2025 OpenHarness Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
