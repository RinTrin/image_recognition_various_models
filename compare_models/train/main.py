
import train
import make_dataframe
import pandas as pd
import model_dic

def main(use_pretrained):
    model_names = list(model_dic.make_model_dic(use_pretrained).keys())
    data = {}
    for model_name in model_names:
        print('\n', model_name, '  NOW READING\n')
        model_data = train.train(model_name, epoch_size=1, pretrained=use_pretrained)
        data[str(model_name)] = model_data
    print(data)
    df = make_dataframe.make_dataframe(data)
    df.head()

if __name__=='__main__':
    main(use_pretrained=True)