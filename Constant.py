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


class Hierarchy:
    SCENARIO = 'Scenario'
    CATEGORY = 'Category'
    TYPE = 'Type'
    SYMBOL = 'Symbol'


class DNNParameters:
    epochStep = {Hierarchy.SCENARIO: 6400, Hierarchy.CATEGORY: 1600,
                 Hierarchy.TYPE: 800, Hierarchy.SYMBOL: 100}

    class LearningRate:
        STEP_1 = 0.01
        STEP_2 = 0.001
        STEP_3 = 0.0001  # base
        STEP_4 = 0.00001
        STEP_5 = 0.000001


class FileLoad:
    STAT_TIME_DELTA_DIR_PATH = 'Dataset/STAT_TIME_DELTA'
    TOP_DIR_PATH = 'Dataset/TOP'
    CS_STAT_TIME_DELTA_DIR_PATH = STAT_TIME_DELTA_DIR_PATH + '/' + CS
    GS_STAT_TIME_DELTA_DIR_PATH = STAT_TIME_DELTA_DIR_PATH + '/' + GS
    CS_TOP_DIR_PATH = TOP_DIR_PATH + '/' + CS
    GS_TOP_DIR_PATH = TOP_DIR_PATH + '/' + GS


class FileSave:
    ML_RESULT = 'ML_Result'
    DNN = 'DNN'
    OTHERS = 'Others'
    CLASSIFICATION_RESULT = 'Classification_Result'
    LOSS_RATE = 'Loss_Rate'
    CR_FILE_NAME = 'result.json'
    LR_FILE_NAME = 'loss_rate.png'

    DNN_CR_CS_STD_CATEGORY = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    DNN_CR_CS_STD_SCENARIO = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    DNN_CR_CS_TOP_CATEGORY = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                              '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    DNN_CR_CS_TOP_SCENARIO = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                              '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    DNN_CR_CS_TOP_SYMBOL = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                            '/' + Hierarchy.SYMBOL + '/' + CR_FILE_NAME)
    DNN_CR_CS_TOP_TYPE = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                          '/' + Hierarchy.TYPE + '/' + CR_FILE_NAME)

    DNN_CR_GS_STD_CATEGORY = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    DNN_CR_GS_STD_SCENARIO = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    DNN_CR_GS_TOP_CATEGORY = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                              '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    DNN_CR_GS_TOP_SCENARIO = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                              '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    DNN_CR_GS_TOP_SYMBOL = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                            '/' + Hierarchy.SYMBOL + '/' + CR_FILE_NAME)
    DNN_CR_GS_TOP_TYPE = (ML_RESULT + '/' + DNN + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                          '/' + Hierarchy.TYPE + '/' + CR_FILE_NAME)

    DNN_LR_CS_STD_CATEGORY = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + CS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.CATEGORY + '/' + LR_FILE_NAME)
    DNN_LR_CS_STD_SCENARIO = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + CS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.SCENARIO + '/' + LR_FILE_NAME)
    DNN_LR_CS_TOP_CATEGORY = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/' + Hierarchy.CATEGORY +
                              '/' + LR_FILE_NAME)
    DNN_LR_CS_TOP_SCENARIO = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/' + Hierarchy.SCENARIO +
                              '/' + LR_FILE_NAME)
    DNN_LR_CS_TOP_SYMBOL = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/' + Hierarchy.SYMBOL +
                            '/' + LR_FILE_NAME)
    DNN_LR_CS_TOP_TYPE = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/' + Hierarchy.TYPE +
                          '/' + LR_FILE_NAME)

    DNN_LR_GS_STD_CATEGORY = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + GS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.CATEGORY + '/' + LR_FILE_NAME)
    DNN_LR_GS_STD_SCENARIO = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + GS + '/' + STAT_TIME_DELTA +
                              '/' + Hierarchy.SCENARIO + '/' + LR_FILE_NAME)
    DNN_LR_GS_TOP_CATEGORY = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + GS + '/' + TOP + '/' + Hierarchy.CATEGORY +
                              '/' + LR_FILE_NAME)
    DNN_LR_GS_TOP_SCENARIO = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + GS + '/' + TOP + '/' + Hierarchy.SCENARIO +
                              '/' + LR_FILE_NAME)
    DNN_LR_GS_TOP_SYMBOL = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + GS + '/' + TOP + '/' + Hierarchy.SYMBOL +
                            '/' + LR_FILE_NAME)
    DNN_LR_GS_TOP_TYPE = (ML_RESULT + '/' + DNN + '/' + LOSS_RATE + '/' + GS + '/' + TOP + '/' + Hierarchy.TYPE +
                          '/' + LR_FILE_NAME)

    # ----------------------------------------------------------------------------------------------------------------

    OTHER_CR_CS_STD_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + CS +
                                '/' + STAT_TIME_DELTA + '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    OTHER_CR_CS_STD_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + CS +
                                '/' + STAT_TIME_DELTA + '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    OTHER_CR_CS_TOP_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                                '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    OTHER_CR_CS_TOP_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                                '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    OTHER_CR_CS_TOP_SYMBOL = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                              '/' + Hierarchy.SYMBOL + '/' + CR_FILE_NAME)
    OTHER_CR_CS_TOP_TYPE = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + CS + '/' + TOP +
                            '/' + Hierarchy.TYPE + '/' + CR_FILE_NAME)

    OTHER_CR_GS_STD_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + GS + '/'
                                + STAT_TIME_DELTA + '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    OTHER_CR_GS_STD_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + GS + '/'
                                + STAT_TIME_DELTA + '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    OTHER_CR_GS_TOP_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                                '/' + Hierarchy.CATEGORY + '/' + CR_FILE_NAME)
    OTHER_CR_GS_TOP_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                                '/' + Hierarchy.SCENARIO + '/' + CR_FILE_NAME)
    OTHER_CR_GS_TOP_SYMBOL = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                              '/' + Hierarchy.SYMBOL + '/' + CR_FILE_NAME)
    OTHER_CR_GS_TOP_TYPE = (ML_RESULT + '/' + OTHERS + '/' + CLASSIFICATION_RESULT + '/' + GS + '/' + TOP +
                            '/' + Hierarchy.TYPE + '/' + CR_FILE_NAME)

    OTHER_LR_CS_STD_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + CS + '/' + STAT_TIME_DELTA +
                                '/' + Hierarchy.CATEGORY + '/' + LR_FILE_NAME)
    OTHER_LR_CS_STD_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + CS + '/' + STAT_TIME_DELTA +
                                '/' + Hierarchy.SCENARIO + '/' + LR_FILE_NAME)
    OTHER_LR_CS_TOP_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/'
                                + Hierarchy.CATEGORY + '/' + LR_FILE_NAME)
    OTHER_LR_CS_TOP_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/'
                                + Hierarchy.SCENARIO + '/' + LR_FILE_NAME)
    OTHER_LR_CS_TOP_SYMBOL = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/'
                              + Hierarchy.SYMBOL + '/' + LR_FILE_NAME)
    OTHER_LR_CS_TOP_TYPE = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + CS + '/' + TOP + '/'
                            + Hierarchy.TYPE + '/' + LR_FILE_NAME)

    OTHER_LR_GS_STD_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + GS + '/' + STAT_TIME_DELTA +
                                '/' + Hierarchy.CATEGORY + '/' + LR_FILE_NAME)
    OTHER_LR_GS_STD_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + GS + '/' + STAT_TIME_DELTA +
                                '/' + Hierarchy.SCENARIO + '/' + LR_FILE_NAME)
    OTHER_LR_GS_TOP_CATEGORY = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + GS + '/' + TOP +
                                '/' + Hierarchy.CATEGORY + '/' + LR_FILE_NAME)
    OTHER_LR_GS_TOP_SCENARIO = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + GS + '/' + TOP +
                                '/' + Hierarchy.SCENARIO + '/' + LR_FILE_NAME)
    OTHER_LR_GS_TOP_SYMBOL = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + GS + '/' + TOP +
                              '/' + Hierarchy.SYMBOL + '/' + LR_FILE_NAME)
    OTHER_LR_GS_TOP_TYPE = (ML_RESULT + '/' + OTHERS + '/' + LOSS_RATE + '/' + GS + '/' + TOP +
                            '/' + Hierarchy.TYPE + '/' + LR_FILE_NAME)


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
