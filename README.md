# diffusion-research

Meta/super repository encapsulating my research on diffusion models and inverse problems

To reproduce results in paper:

- Using a CUDA-enabled device, clone the repository:

```bash
git clone --recurse-submodules https://github.com/bd3dowling/diffusion-research.git
```

- Use [`poetry`](https://python-poetry.org/) to install dependencies into a virtual environment:

```bash
poetry install && poetry shell
```

- Run the scripts:

```bash
python gmm.py
python branin.py
python gmm-plot.py
python gmm-table-maker.py
```

- For the superconductor experiments, see [this repo](https://github.com/bd3dowling/superconductor).
