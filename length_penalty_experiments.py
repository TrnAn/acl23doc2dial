import inference_generation
from utils.preprocessing import get_args
import numpy as np
import pandas as pd
import plotly.express as px
import os

if __name__ == '__main__':
    kwargs = get_args()
    kwargs["cache_dir"] = "0_baseline"
    scores_all_rougel = []
    scores_all_bleu = []
    scores_all_f1 = []
    scores_fr_rougel = []
    scores_fr_bleu = []
    scores_fr_f1 = []
    scores_vi_rougel = []
    scores_vi_bleu = []
    scores_vi_f1 = []
    start = 0.25
    end = 4.
    step = 0.25

    fname = os.path.join(kwargs["cache_dir"], "length_penalty.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname)
    else:
        length_penalties = np.linspace(start, end, num=int((end - start) / step) + 1)

        kwargs["eval_lang"] = [["fr", "vi"], ["fr"], ["vi"]]
        for length_penalty in length_penalties:
            print(f"{length_penalty=}")
            kwargs["length_penalty"] = length_penalty
            
            all_results = inference_generation.main(**kwargs)
            old_fname = os.path.join(kwargs["cache_dir"], f"fr_vi_evaluate_result.json")

            new_fname = os.path.join(kwargs["cache_dir"], f"fr_vi_evaluate_result_lp{length_penalty}.json")
            
            if os.path.exists(old_fname):
                os.rename(old_fname, new_fname)

            scores_all_rougel += [all_results["fr_vi"]["rouge"]]
            scores_all_bleu += [all_results["fr_vi"]["bleu"]]
            scores_all_f1 += [all_results["fr_vi"]["f1"]]

            scores_fr_rougel += [all_results["fr"]["rouge"]]
            scores_fr_bleu += [all_results["fr"]["bleu"]]
            scores_fr_f1 += [all_results["fr"]["f1"]]

            scores_vi_rougel += [all_results["vi"]["rouge"]]
            scores_vi_bleu += [all_results["vi"]["bleu"]]
            scores_vi_f1 += [all_results["vi"]["f1"]]

            # scores_fr += [sum(all_results["fr"].values())]
            # scores_vi += [sum(all_results["vi"].values())]

        # print(scores_all)
        print(scores_all_rougel)
        print(length_penalties)
        df  = pd.DataFrame({'Rouge-L (fr+vi)': scores_all_rougel, 
                            'SacreBLEU (fr+vi)': scores_all_bleu, 
                            'F1-Score (fr+vi)': scores_all_f1,
                            'Rouge-L (fr)': scores_fr_rougel, 
                            'SacreBLEU (fr)': scores_fr_bleu, 
                            'F1-Score (fr)': scores_fr_f1,
                            'Rouge-L (vi)': scores_vi_rougel, 
                            'SacreBLEU (vi)': scores_vi_bleu, 
                            'F1-Score (vi)': scores_vi_f1,
                            'length_penalty': length_penalties
                            })
        
        df.to_csv(os.path.join(kwargs["cache_dir"], "length_penalty.csv"), index=False)

    colors = ['#B19CD9']*3 +  ['#FFA500',]*3 + ['#87AE73']*3
    # fig = px.line(df, x="length_penalty", y=["total_score_fr_vi","total_score_fr", "total_score_vi"], 
    #               color_discrete_sequence=colors)
    print(colors)

    unique_colors = list(set(colors))
    fig = px.line(df, x='length_penalty',
              y=['Rouge-L (fr+vi)', 'SacreBLEU (fr+vi)', 'F1-Score (fr+vi)',
                 'Rouge-L (fr)', 'SacreBLEU (fr)', 'F1-Score (fr)',
                 'Rouge-L (vi)', 'SacreBLEU (vi)', 'F1-Score (vi)'],
              labels={'value': 'Scores', 'Start': 'Start Value'},
              color_discrete_map={
                  'Rouge-L (fr+vi)': unique_colors[0],
                  'SacreBLEU (fr+vi)': unique_colors[0],
                  'F1-Score (fr+vi)': unique_colors[0],
                  'Rouge-L (fr)': unique_colors[1],
                  'SacreBLEU (fr)': unique_colors[1],
                  'F1-Score (fr)': unique_colors[1],
                  'Rouge-L (vi)': unique_colors[2],
                  'SacreBLEU (vi)': unique_colors[2],
                  'F1-Score (vi)': unique_colors[2]
              },
            #   line_dash_map={
            #       'Rouge-L (fr+vi)': 'solid',
            #       'SacreBLEU (fr+vi)': 'dash',
            #       'F1-Score (fr+vi)': 'dot',
            #       'Rouge-L (fr)': 'solid',
            #       'SacreBLEU (fr)': 'dash',
            #       'F1-Score (fr)': 'dot',
            #       'Rouge-L (vi)': 'solid',
            #       'SacreBLEU (vi)': 'dash',
            #       'F1-Score (vi)': 'dot'
            #   }
              )

    fig.update_yaxes(title_text="score") 
    fig.update_layout(legend_title_text="score (eval language)")
    line_dash_styles = ['solid', 'dash', 'dot']*3  # Define the line dash styles for your traces
    for i, trace_name in enumerate(fig.data):
        fig.data[i].line.dash = line_dash_styles[i]


    for idx, column in enumerate(['Rouge-L (fr+vi)', 'SacreBLEU (fr+vi)', 'F1-Score (fr+vi)',
                 'Rouge-L (fr)', 'SacreBLEU (fr)', 'F1-Score (fr)',
                 'Rouge-L (vi)', 'SacreBLEU (vi)', 'F1-Score (vi)']):
        fig.add_trace(px.scatter(df, x="length_penalty", y=column, color_discrete_sequence=[colors[idx]]).data[0])

    fig.write_image(os.path.join(kwargs["cache_dir"], "length_penalty.png"))
    

