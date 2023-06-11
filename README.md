# CS7347

该代码仓库包含课程CS7347用于自然语言推理任务的模型微调和推理的Python脚本以及一个数据增强的Jupyter Notebook。

## 文件结构

- `FineTuneBertDA.py`: 用于对BERT模型进行微调的脚本。
- `FineTuneGPT2.py`: 用于对GPT-2模型进行微调的脚本。
- `InferenceBert.py`: 用于使用微调后的BERT模型进行推理的脚本。
- `InferenceGPT.py`: 用于使用微调后的GPT-2模型进行推理的脚本。
- `notebook-data-augmentation.ipynb`: 一个Jupyter Notebook，演示了数据增强的方法。

## 使用说明

以下是每个脚本的简要说明：

- `FineTuneBertDA.py`：运行此脚本可对BERT模型进行微调，用于特定的自然语言处理任务。请确保在运行之前已安装相关依赖库，并提供相应的训练数据和配置参数。
- `FineTuneGPT2.py`：运行此脚本可对GPT-2模型进行微调，用于特定的自然语言处理任务。请确保在运行之前已安装相关依赖库，并提供相应的训练数据和配置参数。
- `InferenceBert.py`：使用此脚本可进行使用微调后的BERT模型进行推理。请确保已在脚本中指定正确的模型路径和输入数据。
- `InferenceGPT.py`：使用此脚本可进行使用微调后的GPT-2模型进行推理。请确保已在脚本中指定正确的模型路径和输入数据。

对于数据增强的Jupyter Notebook，您可以按照其中的说明和代码演示来使用不同的数据增强方法来增强您的训练数据。

## 依赖库

- 在运行脚本之前，请确保已安装以下依赖库：
  - torch
  - transformers
  - tqdm

## 注意事项

- 本代码仓库仅提供了模型微调和推理的基本示例脚本和Jupyter Notebook，并未包含完整的数据集和模型权重文件。请根据您的实际需求自行准备相应的数据集和权重文件。


