**Model demo** notebook is stored with instructions and visuals: [here](https://github.com/EnronMusk/multi_implicit_cot/blob/main/demo/model_demo.ipynb)

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

Dataset is dynamically generated and saved: [here](https://github.com/EnronMusk/multi_implicit_cot/tree/main/data)

Referenced paper: [here](https://arxiv.org/pdf/2311.01460.pdf)

## **Results** 

Used gpt-2 small (12 layers) and 777k training dataset and 77k test dataset:

|**Model** |**Loss** | **Test** | **Train**|
|----------|----------|----------|----------|
|Teacher|Perplexitity: 1.000465| Test Accuracy: 0.997169| Training Accuracy: 0.999882|
|ThoughtEmulator|Loss: 4.369609| Quasi Test Accuracy 0.977900| Quasi Training Accuracy: 0.977773|
|MindReadingEmulator|Perplexitity: 1.000601| Test Accuracy: 0.996688| Training Accuracy: 0.999745|
|ImplicitStudent|Perplexitity: 1.000000| Test Accuracy: 1.000000| Training Accuracy: 1.000000|

## **Notes** 

The implicit student model performed exceptionally well after being retrained on the train data and its accuracy statistics are accurate to 6 decimal places.

The teacher model can have much better performance with higher eta (learning rate). I multipled eta by **8/5** and saw 0.999831 test accuracy and 0.999995 training, which are both significantly higher. This makes sense because our implicit student performed better than the teacher, when it was trained on the exact same data.
