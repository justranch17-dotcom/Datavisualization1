# Artifact Manifest

This records local project files and the current push policy. It helps keep GitHub focused on source, docs, feedback, and reports while large regenerated data stays local.

## Summary

```text
             push_policy             category  files  total_mb
             do_not_push           local_only  34468 10397.705
             do_not_push        runtime_local      5     0.016
             do_not_push secret_or_credential      1     0.000
do_not_push_github_limit         model_binary      5   817.962
 prefer_artifact_storage       generated_data     44   551.825
 prefer_artifact_storage         model_binary      3    89.967
                 push_ok                other      2     5.909
                 push_ok     source_or_report     70     0.804
```

## Largest Local Files

```text
                                                 path  size_mb       category              push_policy
    model_artifacts\raw_early_structure_models.joblib  249.737   model_binary do_not_push_github_limit
       model_artifacts\entry_bar_timing_models.joblib  181.588   model_binary do_not_push_github_limit
            downloaded_historical_data\BTC_USD_1m.csv  180.047     local_only              do_not_push
   model_artifacts\early_directional_predictor.joblib  170.586   model_binary do_not_push_github_limit
            downloaded_historical_data\ETH_USD_1m.csv  168.573     local_only              do_not_push
model_artifacts\structural_feedback_rich_model.joblib  120.727   model_binary do_not_push_github_limit
       model_artifacts\early_pattern_predictor.joblib   95.324   model_binary do_not_push_github_limit
     model_artifacts\raw_early_structure_features.csv   72.746 generated_data  prefer_artifact_storage
   model_artifacts\structural_day_ensemble_scores.csv   70.939 generated_data  prefer_artifact_storage
       model_artifacts\structural_day_rich_scores.csv   69.387 generated_data  prefer_artifact_storage
            model_artifacts\structural_day_scores.csv   69.318 generated_data  prefer_artifact_storage
               downloaded_historical_data\TSLA_1m.csv   68.801     local_only              do_not_push
          model_artifacts\structural_day_features.csv   68.585 generated_data  prefer_artifact_storage
                downloaded_historical_data\QQQ_1m.csv   65.942     local_only              do_not_push
               downloaded_historical_data\NVDA_1m.csv   65.733     local_only              do_not_push
                downloaded_historical_data\SPY_1m.csv   64.007     local_only              do_not_push
               downloaded_historical_data\AAPL_1m.csv   60.506     local_only              do_not_push
               downloaded_historical_data\PLTR_1m.csv   59.292     local_only              do_not_push
                downloaded_historical_data\AMD_1m.csv   58.661     local_only              do_not_push
                downloaded_historical_data\IWM_1m.csv   57.036     local_only              do_not_push
               downloaded_historical_data\AMZN_1m.csv   54.862     local_only              do_not_push
               downloaded_historical_data\SOFI_1m.csv   54.136     local_only              do_not_push
               downloaded_historical_data\INTC_1m.csv   53.105     local_only              do_not_push
               downloaded_historical_data\MSFT_1m.csv   49.972     local_only              do_not_push
               downloaded_historical_data\META_1m.csv   49.303     local_only              do_not_push
               downloaded_historical_data\COIN_1m.csv   49.194     local_only              do_not_push
              downloaded_historical_data\GOOGL_1m.csv   48.516     local_only              do_not_push
                downloaded_historical_data\TLT_1m.csv   47.364     local_only              do_not_push
                downloaded_historical_data\SLV_1m.csv   45.452     local_only              do_not_push
               downloaded_historical_data\GOOG_1m.csv   45.073     local_only              do_not_push
                 downloaded_historical_data\MU_1m.csv   44.785     local_only              do_not_push
                downloaded_historical_data\GLD_1m.csv   43.382     local_only              do_not_push
                downloaded_historical_data\PFE_1m.csv   42.497     local_only              do_not_push
                downloaded_historical_data\FXI_1m.csv   42.032     local_only              do_not_push
                downloaded_historical_data\VOO_1m.csv   41.175     local_only              do_not_push
                downloaded_historical_data\BAC_1m.csv   40.797     local_only              do_not_push
               downloaded_historical_data\AVGO_1m.csv   40.554     local_only              do_not_push
                downloaded_historical_data\OXY_1m.csv   40.516     local_only              do_not_push
               downloaded_historical_data\PYPL_1m.csv   40.505     local_only              do_not_push
     model_artifacts\structural_feedback_model.joblib   40.500   model_binary  prefer_artifact_storage
```