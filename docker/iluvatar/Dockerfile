# Build from the repository root (same base/release/dev/ci structure as TE-FL):
# docker build --target ci -f docker/iluvatar/Dockerfile \
#   --build-arg VCS_REF=$(git rev-parse HEAD) \
#   -t harbor.baai.ac.cn/flagos-dev/megatron-lm-fl:1dc1e01-iluvatar-dev .
ARG ILUVATAR_BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/transformerengine-fl:f886262-iluvatar-dev
FROM ${ILUVATAR_BASE_IMAGE} AS base
ARG ILUVATAR_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG SOURCE_URL=https://github.com/flagos-ai/Megatron-LM-FL
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG YQ_DOWNLOAD_URL=https://github.com/mikefarah/yq/releases/download/v4.45.1/yq_linux_amd64
LABEL org.opencontainers.image.title="megatron-lm-fl" \
      org.opencontainers.image.description="Iluvatar BI-V150 CoreX 4.5 Megatron environment" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ILUVATAR_BASE_IMAGE}" \
      io.flagos.accelerator.backend="iluvatar" \
      io.flagos.image.stage="base" \
      io.flagos.torch.version="2.10.0+corex.4.5.0.20260804" \
      io.flagos.driver.version="4.5.0"
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/te-ci \
    PATH=/opt/te-ci/bin:/usr/local/corex/bin:${PATH} \
    CI_PYTHON_BIN=/opt/te-ci/bin/python3 \
    GEMS_VENDOR=iluvatar \
    TE_FL_SKIP_CUDA=1 \
    TE_FL_PREFER=flagos \
    TE_FL_PER_OP="rmsnorm_bwd_add=impl:vendor.iluvatar;te_general_grouped_gemm=impl:vendor.iluvatar;multi_tensor_sgd=impl:reference.torch" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVTE_FRAMEWORK=pytorch \
    NVTE_WITH_NCCL_EP=0 \
    PIP_CONSTRAINT=/opt/megatron-image/constraints.txt
RUN mkdir -p /opt/megatron-image \
    && printf '%s\n' 'numpy==1.26.4' 'scipy==1.15.3' \
       'torch==2.10.0+corex.4.5.0.20260804' \
       'torchvision==0.25.0+corex.4.5.0.20260804' \
       'triton==3.2.0+corex.4.5.0.20260804' \
       'ixte==5.0.0+corex.4.5.0.20260804' 'flag-gems==5.4.0' \
       > /opt/megatron-image/constraints.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git gettext-base \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 2 --connect-timeout 15 --max-time 180 \
       "$YQ_DOWNLOAD_URL" -o /usr/local/bin/yq \
    && echo '654d2943ca1d3be2024089eb4f270f4070f491a0610481d128509b2834870049  /usr/local/bin/yq' | sha256sum -c - \
    && chmod 0755 /usr/local/bin/yq && yq --version
# Validate metadata without initializing GPU drivers during the image build.
RUN python3 -c 'import importlib.metadata as m; expected={"torch":"2.10.0+corex.4.5.0.20260804","torchvision":"0.25.0+corex.4.5.0.20260804","triton":"3.2.0+corex.4.5.0.20260804","ixte":"5.0.0+corex.4.5.0.20260804","flag_gems":"5.4.0"}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print(actual)'
# Generate the guarded patch helper inside the image; keep a single Dockerfile.
RUN printf '%s\n' \
    "\"\"\"Apply the validated reference SGD fix without replacing native TE libraries.\"\"\"" \
    "" \
    "import hashlib" \
    "import sysconfig" \
    "from pathlib import Path" \
    "" \
    "ORIGINAL_SHA256 = \"eba745d33febe3a65a31e684a75010acd24e758ff26aefc0ace51aad40e9f8f1\"" \
    "PATCHED_SHA256 = \"9fcf24b796253f514495c2b9b29598422660d402cec2c17e16411242ba112ca8\"" \
    "FUNCTION = '''@torch.no_grad()" \
    "def multi_tensor_sgd_torch(" \
    "    chunk_size: int," \
    "    noop_flag: torch.Tensor," \
    "    tensor_lists: List[List[torch.Tensor]]," \
    "    wd: float," \
    "    momentum: float," \
    "    dampening: float," \
    "    lr: float," \
    "    nesterov: bool," \
    "    first_run: bool," \
    "    wd_after_momentum: bool," \
    "    scale: float," \
    ") -> None:" \
    "    \"\"\"PyTorch implementation of the TEX multi_tensor_sgd kernel contract.\"\"\"" \
    "    if noop_flag.item() != 0:" \
    "        return" \
    "    if len(tensor_lists) not in (3, 4):" \
    "        raise ValueError(\"Expected [grads, params, momentum_buffers, optional model_params]\")" \
    "    grads, params, momentum_buffers = tensor_lists[:3]" \
    "    if any(len(items) != len(params) for items in tensor_lists):" \
    "        raise ValueError(\"All tensor lists must have the same length\")" \
    "    for i, (grad, param, buf) in enumerate(zip(grads, params, momentum_buffers)):" \
    "        if grad is None:" \
    "            continue" \
    "        # Match the native kernel's FP32 intermediate arithmetic, including" \
    "        # half parameters and optional master-to-model parameter copies." \
    "        update = grad.float().mul(scale)" \
    "        weight = param.float()" \
    "        if wd != 0 and not wd_after_momentum:" \
    "            update = update.add(weight, alpha=wd)" \
    "        if momentum != 0:" \
    "            if first_run:" \
    "                next_momentum = update" \
    "            else:" \
    "                next_momentum = buf.float().mul(momentum).add(update, alpha=1 - dampening)" \
    "            buf.copy_(next_momentum)" \
    "            if nesterov:" \
    "                update = update.add(next_momentum, alpha=momentum)" \
    "            else:" \
    "                update = next_momentum" \
    "        if wd != 0 and wd_after_momentum:" \
    "            update = update.add(weight, alpha=wd)" \
    "        param.copy_(weight.add(update, alpha=-lr))" \
    "        if len(tensor_lists) == 4:" \
    "            tensor_lists[3][i].copy_(param)" \
    "" \
    "" \
    "'''" \
    "" \
    "" \
    "def main():" \
    "    path = Path(sysconfig.get_path(\"purelib\")) / (" \
    "        \"transformer_engine/plugin/core/backends/reference/impl/optimizer.py\"" \
    "    )" \
    "    source = path.read_text()" \
    "    digest = hashlib.sha256(source.encode()).hexdigest()" \
    "    if digest == PATCHED_SHA256:" \
    "        print(\"TE-FL reference SGD patch already applied\")" \
    "        return" \
    "    if digest != ORIGINAL_SHA256:" \
    "        raise RuntimeError(f\"Unrecognized TE-FL optimizer source {digest}; review patch compatibility\")" \
    "    start = source.index(\"def multi_tensor_sgd_torch(\")" \
    "    end = source.index(\"def multi_tensor_compute_scale_and_scale_inv_torch(\", start)" \
    "    patched = source[:start] + FUNCTION + source[end:]" \
    "    if hashlib.sha256(patched.encode()).hexdigest() != PATCHED_SHA256:" \
    "        raise RuntimeError(\"TE-FL patch differs from the validated implementation\")" \
    "    compile(patched, str(path), \"exec\")" \
    "    path.with_suffix(\".py.before-megatron-sgd-fix\").write_text(source)" \
    "    path.write_text(patched)" \
    "    print(\"Applied validated TE-FL reference SGD patch\")" \
    "" \
    "" \
    "if __name__ == \"__main__\":" \
    "    main()" > /opt/megatron-image/apply_te_fl_patches.py
COPY .github/scripts/set_env_common.sh /opt/megatron-image/set_env_common.sh
RUN python3 /opt/megatron-image/apply_te_fl_patches.py \
    && bash -c 'source /opt/megatron-image/set_env_common.sh; ci_install_uv_compatibility_shim true'
WORKDIR /workspace/Megatron-LM-FL
COPY . .
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation .
CMD ["bash"]

FROM base AS release
LABEL io.flagos.image.stage="release"
WORKDIR /workspace/Megatron-LM-FL

FROM base AS dev
LABEL io.flagos.image.stage="dev"
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python3 -m pip install --no-cache-dir --no-deps \
    pybind11==3.1.0 ninja==1.13.2 mock==5.2.0 \
    pytest==8.4.2 coverage==7.17.0 pytest-cov==7.1.0 \
    pytest-mock==3.17.0 pytest-asyncio==1.4.1.post1 pytest-random-order==1.2.0 \
    py-spy==0.4.2

FROM dev AS ci
LABEL io.flagos.image.stage="ci"
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG EMERGING_OPTIMIZERS_SOURCE=git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@1effa026ff096b7fa1063ca2fba19d98be6e6cdf
# Supplemental packages from the tested container. Keep all native vendor
# packages inherited; no CUDA wheels or replacement Torch/TE/IXTE are installed.
# PuLP<4 and the two missing CPU dependencies repair pip-check conflicts.
RUN python3 -m pip install --no-cache-dir --no-deps \
    Deprecated==3.0.1 PuLP==3.3.2 \
    aiohappyeyeballs==2.7.1 aiohttp==3.14.5 aiosignal==1.4.0 \
    annotated-doc==0.0.5 anyio==4.16.0 asciitree==1.0.0 attrs==26.1.0 \
    click==8.5.0 cloudpickle==3.1.2 cpsat-logutils==1.0.0 cwevals==0.0.1b1 \
    defusedxml==0.7.1 distro==1.9.0 fastapi==0.162.0 fasteners==0.20 \
    frozenlist==1.8.0 googleapis-common-protos==1.76.0 grpcio-tools==1.85.0 \
    h11==0.16.0 httpcore==1.0.9 httpcore2==2.16.0 httpx==0.28.1 httpx2==2.16.0 \
    huggingface-hub==0.26.5 jiter==0.18.0 joblib==1.6.0 \
    markdown-it-py==4.2.0 mdurl==0.1.2 multidict==7.2.1 nltk==3.10.4 \
    numcodecs==0.15.2 nvidia-ml-py==13.615.71 nvidia-modelopt==0.47.0 \
    nvidia-resiliency-ext==0.6.0 omegaconf==2.4.0 openai==3.42.1 \
    opentelemetry-api==1.45.2 opentelemetry-exporter-http-transport==0.66b2 \
    opentelemetry-exporter-otlp-common==0.66b2 \
    opentelemetry-exporter-otlp-proto-common==1.45.2 \
    opentelemetry-exporter-otlp-proto-http==1.45.2 opentelemetry-proto==1.45.2 \
    opentelemetry-sdk==1.45.2 opentelemetry-semantic-conventions==0.66b2 \
    orloge==1.0.0 platformdirs==4.14.3 propcache==0.5.4 rich==15.0.0 \
    sentencepiece==0.2.2 sniffio==1.3.1 starlette==1.10.0 tensorstore==0.1.86 \
    tiktoken==0.14.0 tokenizers==0.20.3 transformers==4.46.3 truststore==0.10.5 \
    uvicorn==0.54.0 wandb==0.31.0 wrapt==2.5.2 xxhash==4.0.2 yarl==1.25.1 zarr==2.18.7 \
    && python3 -m pip install --no-cache-dir --no-deps --no-build-isolation "$EMERGING_OPTIMIZERS_SOURCE" \
    && python3 -m pip check \
    && python3 -c 'import importlib.metadata as m; assert m.version("emerging-optimizers")=="0.2.0"; print("CI dependencies validated")'
# Match the real offline GPT2 cache used by data preprocessing tests.
ARG GPT2_CACHE_URL=
RUN if [ -n "$GPT2_CACHE_URL" ]; then \
      mkdir -p /root/.cache/huggingface/hub \
      && curl --fail --location --retry 2 "$GPT2_CACHE_URL" | tar -xz -C /root/.cache/huggingface/hub; \
    else \
      python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e", allow_patterns=["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"])'; \
    fi
WORKDIR /workspace/Megatron-LM-FL

# Apex is included for Iluvatar adaptation diagnostics. The LayerNorm extension
# compiles, but its forward/backward numerical checks currently fail on BI-V150.
# Keep the existing sequence_parallel test exclusions until this is resolved.
ARG APEX_SOURCE_URL=https://codeload.github.com/NVIDIA/apex/tar.gz/575968bc1f9127ccd61003a681472a83af4ff1a1
ARG APEX_SOURCE_SHA256=3bed9320c5811afaef780fbd2eca024229f340c207a13abb091d132cd9df17dc
RUN mkdir -p /opt/megatron-image/apex-source \
    && curl --fail --location --retry 2 --connect-timeout 15 --max-time 180 \
       "$APEX_SOURCE_URL" -o /opt/megatron-image/apex-source.tar.gz \
    && test -n "$APEX_SOURCE_SHA256" \
    && echo "$APEX_SOURCE_SHA256  /opt/megatron-image/apex-source.tar.gz" | sha256sum -c - \
    && tar -xzf /opt/megatron-image/apex-source.tar.gz --strip-components=1 \
       -C /opt/megatron-image/apex-source \
    && cd /opt/megatron-image/apex-source \
    && env -u APEX_CPP_EXT -u APEX_CUDA_EXT CUDA_HOME=/usr/local/corex \
       python3 -m pip install --no-cache-dir --no-deps --no-build-isolation --no-index . \
    && printf '%s\n' \
       'from setuptools import setup' \
       'from torch.utils.cpp_extension import BuildExtension, CUDAExtension' \
       'setup(name="apex-norm-probe", ext_modules=[CUDAExtension(name="fused_layer_norm_cuda", sources=["csrc/layer_norm_cuda.cpp", "csrc/layer_norm_cuda_kernel.cu"], extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-x", "ivcore"]})], cmdclass={"build_ext": BuildExtension})' \
       > setup_norm_probe.py \
    && CUDA_HOME=/usr/local/corex MAX_JOBS=2 python3 setup_norm_probe.py build_ext --inplace \
    && python3 -c 'import glob, shutil, sysconfig; artifacts=glob.glob("fused_layer_norm_cuda*.so"); assert len(artifacts)==1, artifacts; shutil.copy2(artifacts[0], sysconfig.get_path("platlib"))' \
    && cd /workspace/Megatron-LM-FL \
    && python3 -c 'import apex, fused_layer_norm_cuda; from apex.normalization import FusedLayerNorm; print("Apex import validated; Iluvatar LayerNorm numerical validation remains failing")' \
    && python3 -m pip check \
    && rm -rf /opt/megatron-image/apex-source/build
LABEL io.flagos.apex.revision="575968bc1f9127ccd61003a681472a83af4ff1a1" \
      io.flagos.apex.layernorm.numerical-validation="failed"
