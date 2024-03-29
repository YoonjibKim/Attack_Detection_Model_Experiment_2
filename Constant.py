CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Correct_ID_Random_CS_Off_Gaussian_Off'
CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON = 'Correct_ID_Random_CS_Off_Gaussian_On'
CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF = 'Correct_ID_Random_CS_On_Gaussian_Off'
CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON = 'Correct_ID_Random_CS_On_Gaussian_On'
WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Wrong_CS_TS_Random_CS_Off_Gaussian_Off'
WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON = 'Wrong_CS_TS_Random_CS_Off_Gaussian_On'
WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF = 'Wrong_CS_TS_Random_CS_On_Gaussian_Off'
WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON = 'Wrong_CS_TS_Random_CS_On_Gaussian_On'
WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Wrong_EV_TS_Random_CS_Off_Gaussian_Off'
WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON = 'Wrong_EV_TS_Random_CS_Off_Gaussian_On'
WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF = 'Wrong_EV_TS_Random_CS_On_Gaussian_Off'
WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON = 'Wrong_EV_TS_Random_CS_On_Gaussian_On'
WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF = 'Wrong_ID_Random_CS_Off_Gaussian_Off'
WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON = 'Wrong_ID_Random_CS_Off_Gaussian_On'
WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF = 'Wrong_ID_Random_CS_On_Gaussian_Off'
WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON = 'Wrong_ID_Random_CS_On_Gaussian_On'

CID_RCOFF_GOFF = 'cid_rcoff_goff'
CID_RCOFF_GON = 'cid_rcoff_gon'
CID_RCON_GOFF = 'cid_rcon_goff'
CID_RCON_GON = 'cid_rcon_gon'
WCT_RCOFF_GOFF = 'wct_rcoff_goff'
WCT_RCOFF_GON = 'wct_rcoff_gon'
WCT_RCON_GOFF = 'wct_rcon_goff'
WCT_RCON_GON = 'wct_rcon_gon'
WET_RCOFF_GOFF = 'wet_rcoff_goff'
WET_RCOFF_GON = 'wet_rcoff_gon'
WET_RCON_GOFF = 'wet_rcon_goff'
WET_RCON_GON = 'wet_rcon_gon'
WID_RCOFF_GOFF = 'wid_rcoff_goff'
WID_RCOFF_GON = 'wid_rcoff_gon'
WID_RCON_GOFF = 'wid_rcon_goff'
WID_RCON_GON = 'wid_rcon_go'

SCENARIO_NAME_DICT = {CID_RCOFF_GOFF: CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF,
                      CID_RCOFF_GON: CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON,
                      CID_RCON_GOFF: CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF,
                      CID_RCON_GON: CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON,
                      WCT_RCOFF_GOFF: WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF,
                      WCT_RCOFF_GON: WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON,
                      WCT_RCON_GOFF: WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF,
                      WCT_RCON_GON: WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON,
                      WET_RCOFF_GOFF: WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF,
                      WET_RCOFF_GON: WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON,
                      WET_RCON_GOFF: WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF,
                      WET_RCON_GON: WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON,
                      WID_RCOFF_GOFF: WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF,
                      WID_RCOFF_GON: WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON,
                      WID_RCON_GOFF: WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF,
                      WID_RCON_GON: WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON}

CS = 'CS'
GS = 'GS'

STAT_TIME_DELTA = 'STAT_TIME_DELTA'
TOP = 'TOP'

TRAINING_FEATURE = 'training_feature'
TRAINING_LABEL = 'training_label'
TESTING_FEATURE = 'testing_feature'
TESTING_LABEL = 'testing_label'

COMMON = 'common'
EXCLUSIVE = 'exclusive'
ALL = 'all'

DUMMY_DATA = -1.0
ATTACK_LABEL = 1
NORMAL_LABEL = 0

CLASSIFICATION_REPORT = 'CLASSIFICATION_REPORT'
EPOCHS = 'EPOCHS'
LEARNING_RATE = 'LEARNING_RATE'

HIERARCHY = 'HIERARCHY'

F1_SCORE = 'f1-score'
WEIGHTED_AVG = 'weighted avg'
SUPPORT = 'support'
HIGHEST_F1 = 'HIGHEST_F1'
F1_AVERAGE = 'F1_AVERAGE'
SUPPORT_AVERAGE = 'SUPPORT_AVERAGE'
FEATURE_TYPE = 'FEATURE_TYPE'
ML_TYPE = 'ML_TYPE'
TYPE_NAME = 'TYPE_NAME'
CATEGORY = 'CATEGORY'
TYPE = 'TYPE'


class DNNParameters:
    class LearningRate:
        STEP_1 = 0.01
        STEP_2 = 0.001
        STEP_3 = 0.0001  # base
        STEP_4 = 0.00001
        STEP_5 = 0.000001


class LOADING_DB:
    STAT_TIME_DELTA_DIR_PATH = 'Dataset/STAT_TIME_DELTA'
    TOP_DIR_PATH = 'Dataset/TOP'
    CS_STAT_TIME_DELTA_DIR_PATH = STAT_TIME_DELTA_DIR_PATH + '/' + CS
    GS_STAT_TIME_DELTA_DIR_PATH = STAT_TIME_DELTA_DIR_PATH + '/' + GS
    CS_TOP_DIR_PATH = TOP_DIR_PATH + '/' + CS
    GS_TOP_DIR_PATH = TOP_DIR_PATH + '/' + GS


class CLASSIFICATION_RESULT_FILE_PATH:
    DNN_CS_STD_INIT_MATCH = 'ML_Result/DNN/Classification_Result/CS/STAT_TIME_DELTA/Initial_Match/Result.json'
    DNN_CS_TOP_INIT_MATCH = 'ML_Result/DNN/Classification_Result/CS/TOP/Initial_Match/Result.json'
    DNN_GS_STD_INIT_MATCH = 'ML_Result/DNN/Classification_Result/GS/STAT_TIME_DELTA/Initial_Match/Result.json'
    DNN_GS_TOP_INIT_MATCH = 'ML_Result/DNN/Classification_Result/GS/TOP/Initial_Match/Result.json'

    OTHER_CS_STD_INIT_MATCH = 'ML_Result/Others/Classification_Result/CS/STAT_TIME_DELTA/Initial_Match/Result.json'
    OTHER_CS_TOP_INIT_MATCH = 'ML_Result/Others/Classification_Result/CS/TOP/Initial_Match/Result.json'
    OTHER_GS_STD_INIT_MATCH = 'ML_Result/Others/Classification_Result/GS/STAT_TIME_DELTA/Initial_Match/Result.json'
    OTHER_GS_TOP_INIT_MATCH = 'ML_Result/Others/Classification_Result/GS/TOP/Initial_Match/Result.json'


class LOSS_RATE_DIR_PATH:
    DNN_CS_TOP_INIT_MATCH = 'ML_Result/DNN/Loss_Rate/CS/TOP/Initial_Match'
    DNN_GS_TOP_INIT_MATCH = 'ML_Result/DNN/Loss_Rate/GS/TOP/Initial_Match'

    DNN_CS_STD_INIT_MATCH = 'ML_Result/DNN/Loss_Rate/CS/STAT_TIME_DELTA/Initial_Match'
    DNN_GS_STD_INIT_MATCH = 'ML_Result/DNN/Loss_Rate/GS/STAT_TIME_DELTA/Initial_Match'


class MLType:
    ADA_BOOST = 'ADA_BOOST'
    AGGLOMERATIVE_CLUSTERING = 'AGGLOMERATIVE_CLUSTERING'
    DBSCAN = 'DBSCAN'
    DECISION_TREE = 'DECISION_TREE'
    GAUSSIAN_MIXTURE = 'GAUSSIAN_MIXTURE'
    GAUSSIAN_NB = 'GAUSSIAN_NB'
    GRADIENT_BOOST = 'GRADIENT_BOOST'
    KMEANS = 'KMEANS'
    KNN = 'KNN'
    LINEAR_REGRESSION = 'LINEAR_REGRESSION'
    LASSO = 'LASSO'
    RIDGE = 'RIDGE'
    ELASTIC = 'ELASTIC'
    LOGISTIC_REGRESSION = 'LOGISTIC_REGRESSION'
    RANDOM_FOREST = 'RANDOM_FOREST'
    SVM = 'SVM'
    DNN = 'DNN'