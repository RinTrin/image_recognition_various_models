import pandas as pd
import os
def make_dataframe(dic_data):
    """
    dic_data's composition

    {model_name:{epoch:{'train':train_data, 'test':test_data}}}
                ↓

         | model1 | model2 | model3 | ...
         ---------------------------- ...
    train|  acc1  |  acc2  |  acc3  | ...
    test |  acc4  |  acc5  |  acc6  | ...

    ** acc is highest score

    """
    df = pd.DataFrame(index=['train', 'test'])
    for model_name, model_data in dic_data.items():
        use_epoch = list(model_data.keys())[-1]
        if use_epoch == 'highest_epoch':
            for key, value in model_data[use_epoch].items():
                try:
                    df['train'][model_name] = model_data[key]['train']
                except:
                    print(model_data)
                    print(key)
                    print(type(key))
                    print(value)
                    raise ValueError
                df['test'][model_name] = value
        else:
            df['train'][model_name] = model_data[use_epoch]['train']
            df['test'][model_name] = model_data[use_epoch]['test']
        
        print(f'{model_name} : {use_epoch}')
    
    os.makedirs('./save_csv_dir/', exist_ok=True)
    csv_path = './save_csv_dir/models_acc_data.csv'
    df.to_csv(csv_path)