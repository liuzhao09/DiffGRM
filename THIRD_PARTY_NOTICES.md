# Third-Party Notices

This document identifies third-party material included in or used by DiffGRM and clarifies the scope of the repository license.

## RPG

DiffGRM contains source code derived from and modified from:

- Project: **RPG: Generating Long Semantic IDs in Parallel for Recommendation**
- Source: <https://github.com/facebookresearch/RPG_KDD2025>
- Copyright: Copyright © Meta Platforms, Inc. and affiliates
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- Upstream license: <https://github.com/facebookresearch/RPG_KDD2025/blob/main/LICENSE.md>

The original copyright and license notices have been retained in the corresponding source files. DiffGRM includes modifications and newly developed components for diffusion-based Semantic-ID generation. Those modifications and components are copyright © 2025–2026 Kuaishou Technology and are distributed under the repository's CC BY-NC 4.0 license unless otherwise noted.

The references to RPG above provide attribution and identify changes; they do not imply endorsement of DiffGRM by Meta Platforms, Inc. or the RPG authors.

## External Data and Models

DiffGRM can download or use material that is not included in this repository, including:

- Amazon Reviews 2014 data hosted by Stanford SNAP;
- `sentence-transformers/sentence-t5-base` and associated model files;
- Python packages listed in `requirements.txt`.

These materials are not relicensed by DiffGRM. Their respective copyright notices, licenses, access conditions, and terms of use continue to apply. Users are responsible for obtaining the material from its official source and determining whether their intended use is permitted.

## Generated Artifacts

Unless a specific release states otherwise, checkpoints, tokenizer or OPQ caches, and Semantic-ID mappings released by Kuaishou Technology as part of DiffGRM are licensed under CC BY-NC 4.0 only to the extent that Kuaishou Technology holds the relevant rights. No permission is granted here for third-party material contained in, used to create, or required to use those artifacts.
