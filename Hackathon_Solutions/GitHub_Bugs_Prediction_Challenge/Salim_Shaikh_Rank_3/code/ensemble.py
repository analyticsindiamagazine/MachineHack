import numpy as np
import pandas as pd

score_849=pd.read_csv("roberta_base_1epochunfreeze_3epochsfreeze_lr2e5.csv")

score_84944=pd.read_csv("roberta_base_1epochunfreeze_1epochfreeze_3_lr2e5.csv")

score_85022=pd.read_csv("roberta_base_1epochunfreeze_2epochsfreeze_3_lr2e5.csv")

output_df=pd.DataFrame(columns=['label'])

output_df['label']=np.where((score_85022['label']==score_84944['label']),score_85022['label'],
                  (np.where((score_849['label']==score_84944['label']),score_84944['label'],
                  (np.where((score_849['label']==score_85022['label']),score_85022['label'],
                  (np.where((score_849['label']==score_84944['label']) & (score_85022['label']==score_84944['label']) & (score_85022['label']==score_849['label']), score_849['label'],score_85022['label'] )))))))


output_df.to_csv("final_submission.csv",index=False)

