class MjorityClassModel():
    def __init__(self):
        pass

    def fit(self,train_df):
        self.train_df = train_df
        self.labels_list = list(train_df["label"])
        self.majority_label = self.get_majority_class(self.labels_list)

    def get_majority_class(self,labels_list):
        label_count_dict = {}
        for label in labels_list:
            if label not in label_count_dict.keys():
                label_count_dict[label]=1
            else:
                label_count_dict[label]+=1
        max_val = -1
        max_label = -1
        for k,v in label_count_dict.items():
            if max_val < v:
                max_label = k
        return max_label

    def predict(self,val_df):
        length = len(val_df)
        predictions = [self.majority_label]*length
        return predictions

    def score(self,val_df):
        from sklearn.metrics import roc_auc_score, accuracy_score
        y_pred = self.predict(val_df)
        self.base_acc = accuracy_score(self.valdf["label"],y_pred)
        self.base_rocauc = roc_auc_score(self.valdf["label"],y_pred)
        return self.base_acc, self.base_rocauc
