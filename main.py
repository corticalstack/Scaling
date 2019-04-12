"""
==========================================================
Example scaling techniques using KDD Cup 1999 IDS dataset
==========================================================

The following examples demonstrate various scaling techniques
for a dataset in which classes are extremely imbalanced with heavily skewed features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from collections import OrderedDict
import itertools
import warnings


class Model:
    def __init__(self):
        self.enabled = False
        self.X_train = None
        self.y_train = None
        self.random_state = 20
        self.predictions = None
        self.base = {'model': None,
                     'stext': None,
                     'scores': None,
                     'cm': None}

    def fit(self, x, y):
        self.base['model'].fit(x, y)

    def predict(self, x, y):
        return cross_val_predict(self.base['model'], x, y, cv=10)


class RandomForestClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.base['stext'] = 'RFC'
        self.base['model'] = RandomForestClassifier(random_state=self.random_state)


class XgboostClf(Model):
    def __init__(self):
        Model.__init__(self)
        self.base['stext'] = 'XGC'
        self.base['model'] = XGBClassifier(random_state=self.random_state)


class Scaling:
    def __init__(self):
        print(__doc__)
        self.dataset = None
        self.x = None
        self.y = None
        self.scores = OrderedDict()
        self.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
        self.columns_scaled = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                               'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                               'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                               'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                               'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                               'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                               'dst_host_srv_rerror_rate']

        self.attack_category_int = [0, 1, 2, 3, 4]
        self.attack_category = ['normal', 'dos', 'u2r', 'r2l', 'probe']

        # Load data then set column names
        self.load_data()
        self.set_columns()

        # Drop large number of duplicates in dataset
        self.drop_duplicates()

        # Clean up the label column data
        self.clean()

        # Set binary target label
        self.set_binary_label()

        # Set attack_category to more clearly see the majority/minority classes - there are 5 "classes"
        self.set_attack_category()

        # Deal with outliers
        self.outliers()

        self.set_x_y(self.dataset)

        for scaler in (StandardScaler(),
                       Normalizer(),
                       MinMaxScaler(feature_range=(0, 1)),
                       Binarizer(threshold=0.0),
                       RobustScaler(quantile_range=(25, 75)),
                       PowerTransformer(method='yeo-johnson'),
                       QuantileTransformer(output_distribution='normal'),
                       QuantileTransformer(output_distribution='uniform')):
            self.scale(scaler)

        self.show_scores()

    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(fname='plots/' + 'CM - ' + title, dpi=300, format='png')
        plt.show()

    def load_data(self):
        self.dataset = pd.read_csv('kddcup.data_10_percent')
        print('--- Original Shape')
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def set_columns(self):
        self.dataset.columns = self.columns

    def drop_duplicates(self):
        print('\n--- Shape after duplicated dropped')
        self.dataset.drop_duplicates(keep='first', inplace=True)
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def clean(self):
        self.dataset['label'] = self.dataset['label'].str.rstrip('.')

    def set_binary_label(self):
        conditions = [
            (self.dataset['label'] == 'normal'),
            (self.dataset['label'] == 'back') | (self.dataset['label'] == 'buffer_overflow') |
            (self.dataset['label'] == 'ftp_write') | (self.dataset['label'] == 'guess_passwd') |
            (self.dataset['label'] == 'imap') | (self.dataset['label'] == 'ipsweep') |
            (self.dataset['label'] == 'land') | (self.dataset['label'] == 'loadmodule') |
            (self.dataset['label'] == 'multihop') | (self.dataset['label'] == 'neptune') |
            (self.dataset['label'] == 'nmap') | (self.dataset['label'] == 'perl') |
            (self.dataset['label'] == 'phf') | (self.dataset['label'] == 'pod') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'rootkit') |
            (self.dataset['label'] == 'satan') | (self.dataset['label'] == 'smurf') |
            (self.dataset['label'] == 'spy') | (self.dataset['label'] == 'teardrop') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster')
        ]
        choices = [0, 1]
        self.dataset['target'] = np.select(conditions, choices, default=0)

    def set_attack_category(self):
        conditions = [
            (self.dataset['label'] == 'normal'),
            (self.dataset['label'] == 'back') | (self.dataset['label'] == 'land') |
            (self.dataset['label'] == 'neptune') | (self.dataset['label'] == 'pod') |
            (self.dataset['label'] == 'smurf') | (self.dataset['label'] == 'teardrop'),
            (self.dataset['label'] == 'buffer_overflow') | (self.dataset['label'] == 'loadmodule') |
            (self.dataset['label'] == 'perl') | (self.dataset['label'] == 'rootkit'),
            (self.dataset['label'] == 'ftp_write') | (self.dataset['label'] == 'guess_passwd') |
            (self.dataset['label'] == 'imap') | (self.dataset['label'] == 'multihop') |
            (self.dataset['label'] == 'phf') | (self.dataset['label'] == 'spy') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster'),
            (self.dataset['label'] == 'ipsweep') | (self.dataset['label'] == 'nmap') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'satan')
        ]
        self.dataset['attack_category'] = np.select(conditions, self.attack_category, default='na')
        self.dataset['attack_category_int'] = np.select(conditions, self.attack_category_int, default=0)

    def set_x_y(self, ds):
        self.x = ds.iloc[:, :-4]
        self.y = ds.iloc[:, -1].values

    def scale(self, scaler):
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15, 8))
        x = self.x[self.columns_scaled]

        # Ignore data conversion warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_x = scaler.fit_transform(x)

        res_x = pd.DataFrame(res_x, columns=self.columns_scaled)

        ax1.set_title('Before Scaling', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        sns.kdeplot(self.x['duration'], ax=ax1)
        sns.kdeplot(self.x['src_bytes'], ax=ax1)
        sns.kdeplot(self.x['dst_bytes'], ax=ax1)
        sns.kdeplot(self.x['serror_rate'], ax=ax1)
        sns.kdeplot(self.x['diff_srv_rate'], ax=ax1)

        ax2.set_title('After ' + scaler.__class__.__name__ + ' Scaler', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        sns.kdeplot(res_x['duration'], ax=ax2)
        sns.kdeplot(res_x['src_bytes'], ax=ax2)
        sns.kdeplot(res_x['dst_bytes'], ax=ax2)
        sns.kdeplot(res_x['serror_rate'], ax=ax2)
        sns.kdeplot(res_x['diff_srv_rate'], ax=ax2)
        plt.savefig(fname='plots/' + 'KDE - ' + scaler.__class__.__name__, dpi=300, format='png')
        plt.show()

        for model in (RandomForestClf(),
                      XgboostClf()):
            model.fit(res_x, self.y)
            y_pred = model.predict(res_x, self.y)
            self.show_cm_multiclass(self.y, y_pred, model.__class__.__name__ + ' - ' + scaler.__class__.__name__)
            self.register_score(scaler, model, res_x, self.y, y_pred)

    def register_score(self, scaler, clf, x, y, y_pred):
        prefix = scaler.__class__.__name__ + '_' + clf.__class__.__name__'_'

        # Warnings caught to suppress issues with minority classes having no predicted label values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + 'recall'] = recall_score(y, y_pred, average=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + 'precision'] = precision_score(y, y_pred, average=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scores[prefix + 'f1'] = f1_score(y, y_pred, average=None)

    def show_scores(self):
        print('--- Prediction Scores')
        for sid, score in self.scores.items():
            print('\nID: {}'.format(sid))
            print('\t\tScore{}'.format(score))

    def show_cm_multiclass(self, y, y_pred, title):
        cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4])
        self.plot_confusion_matrix(cm, classes=[0, 1, 2, 3, 4], title=title)

    def outliers(self):
        print('\n--- Removing extreme outliers')
        self.dataset.describe()
        for col in self.dataset.columns:
            if self.dataset[col].dtype == np.float64 or self.dataset[col].dtype == np.int64:
                threshold = self.dataset[col].max() * 0.95
                outliers = self.dataset[(self.dataset[col] > 50) & (self.dataset[col] > threshold)]
                if (not outliers.empty) and (len(outliers) < (self.dataset.shape[0] * 0.0001)):
                    print('For column {} deleting {} rows over value {}'.format(col, len(outliers), threshold))
                    self.dataset = pd.concat([self.dataset, outliers]).drop_duplicates(keep=False)


scaling = Scaling()
