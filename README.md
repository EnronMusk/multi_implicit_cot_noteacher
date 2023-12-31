This repo is a extentsion of the original work: [here](https://github.com/EnronMusk/multi_implicit_cot)

The idea was to answer the question: _Can we use implicit reasoning without a teacher model?_ The goal was to mimic the same implicit reasoning used by predicting teacher states by instead predicting the pure CoT and to implicity use it in the chain of reasoning. This requires no teacher model, and trains much faster while producing similar results.

**Model demo** notebook is stored with instructions and visuals: [here](https://github.com/EnronMusk/multi_implicit_cot_noteacher/blob/main/demo/model_demo.ipynb)

Simply run the notebook to explore results. The notebook does request user input for a custom prediction, but otherwise its fully automated.

## **Data Format**
The format of training and test datasets follow this format:
``` 
[input 1a] $ [input 1b]||[CoT 1a] $ [CoT 1b] #### [output 1a] $ [output 1b]
[input 2a] $ [input 2b]||[CoT 2a] $ [CoT 2b] #### [output 2a] $ [output 2b]
[input 3a] $ [input 3b]||[CoT 3a] $ [CoT 3b] #### [output 3a] $ [output 3b]
```
Example entry:
``` 
1 7 * 1 3 $ 6 2 * 6 3||1 7 0 + 0 3 1 2 $ 6 5 1 + 0 8 7 0 #### 1 0 2 2 $ 6 3 9 0
```
Each multiplication is delimited by `$`. The `1 7 * 1 3` corresponds to `31 * 71` and `1 7 0 + 0 3 1 2` corresponds to `2130 + 71` and `1 0 2 2 ` corresponds to `2201`

Dataset is dynamically generated and saved: [here](https://github.com/EnronMusk/multi_implicit_cot_noteacher/tree/main/data)

Referenced paper: [here](https://arxiv.org/pdf/2311.01460.pdf)

## **Results** 
Used gpt-2 small (12 layers) and 777k training dataset and 77k test dataset without teacher:

|**Model** |**Loss** | **Test** | **Train**|
|----------|----------|----------|----------|
|ThoughtEmulator|Perplexitity: 1.000000| Test Accuracy 1.000000| Training Accuracy: 1.000000|
|MindReadingEmulator|Perplexitity: 1.187874| Test Accuracy: 1.000000| Training Accuracy: .930459|
|ImplicitStudent|Perplexitity: 1.000026| Test Accuracy: 1.000000| Training Accuracy: 1.000000|

**NOTE** The training accuracy in the mind reading emulator is biased because the model had some slight issues with the context, and predicted 1 extra symbol. Obviously it was accurate enough to extract the answer though so I left it as is.

Used gpt-2 small (12 layers) and 777k training dataset and 77k test dataset with teacher:

|**Model** |**Loss** | **Test** | **Train**|
|----------|----------|----------|----------|
|Teacher|Perplexitity: 1.000465| Test Accuracy: 0.997169| Training Accuracy: 0.999882|
|ThoughtEmulator|Loss: 4.369609| Quasi Test Accuracy 0.977900| Quasi Training Accuracy: 0.977773|
|MindReadingEmulator|Perplexitity: 1.000601| Test Accuracy: 0.996688| Training Accuracy: 0.999745|
|ImplicitStudent|Perplexitity: 1.000000| Test Accuracy: 1.000000| Training Accuracy: 1.000000|
