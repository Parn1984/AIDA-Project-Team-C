from data_ex import gen_xy
from data_ex import scale
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA


# Feature Extraction with RFE
def fea_ex_rfe(my_x, my_y, my_feat):
    # feature extraction
    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select=1)
    fit = rfe.fit(my_x, my_y)
    print('###')
    print('### Feature Extraction with RFE')
    print('###')
    feat_imp = pd.DataFrame(
        {'Feature Name': my_feat,
         'Feature Importance': fit.ranking_
         })
    print(feat_imp.sort_values(by='Feature Importance', ascending=True))


# Feature Extraction with PCA
def fea_ex_pca(my_x):
    # feature extraction
    pca = PCA(n_components=5)
    pca.fit(my_x)
    x_pca = pca.transform(my_x)
    # summarize components
    print('###')
    print('### Feature Extraction with PCA')
    print('###')
    print('Explained Variance: %s' % pca.explained_variance_ratio_)
    print(x_pca.shape)


# Feature Importance with Extra Trees Classifier
def fea_im_etc(my_x, my_y, my_feat):
    # feature extraction
    dtc = DecisionTreeClassifier()
    dtc.fit(my_x, my_y)

    etc = ExtraTreesClassifier(n_estimators=10)
    etc.fit(my_x, my_y)

    rfc = RandomForestClassifier()
    rfc.fit(my_x, my_y)

    abc = AdaBoostClassifier()
    abc.fit(my_x, my_y)

    pd.options.display.max_columns = None
    feat_imp = pd.DataFrame(
        {'Importance DTC': dtc.feature_importances_,
         'Importance RFC': rfc.feature_importances_,
         'Importance ABC': abc.feature_importances_,
         'Importance ETC': etc.feature_importances_},
        index=[my_feat])

    print('###')
    print('### Feature Importance with Extra Trees Classifier')
    print('###')
    print('DTC = DecisionTreeClassifier')
    print('RFC = RandomForestClassifier')
    print('ABC = AdaBoostClassifier')
    print('ETC = ExtraTreesClassifier')
    print(feat_imp)
    return feat_imp


if __name__ == '__main__':
    """
    Test the script
    """
    from sklearn.preprocessing import StandardScaler

    data_file = '../../data/churn.csv'
    x, y = gen_xy(data_file)
    feat = x.columns

    std = StandardScaler()

    sca_columns = ['account_length',
                   'number_customer_service_calls',
                   'number_vmail_messages',
                   'total_day_minutes',
                   'total_day_calls',
                   'total_day_charge',
                   'total_eve_minutes',
                   'total_eve_calls',
                   'total_eve_charge',
                   'total_night_minutes',
                   'total_night_calls',
                   'total_night_charge',
                   'total_intl_minutes',
                   'total_intl_calls',
                   'total_intl_charge']

    std.fit(x[sca_columns])
    x = scale(x, std, sca_columns)
    fea_ex_rfe(x, y, feat)
    fea_ex_pca(x)
    fea_im_etc(x, y, feat)
