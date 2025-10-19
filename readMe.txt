How to run the codes
-streamlit_alpha_av.py
streamlit run streamlit_alpha_av.py

-equity_processing_pipeline.py
python equity_processing_pipeline.py --input "input_data/COP (Conoco Phillips).csv" --output processing_output/cop_processed.csv

-fx_processing_pipeline.py
python fx_processing_pipeline.py -i input_data/USD_EUR_daily_full.csv -o processing_output/fx_processed.csv

-option_processing_pipeline.py
python option_processing_pipeline.py -i input_data/options.csv -o processing_output/option_processed.csv

-equity_feature_engineering.py
python equity_feature_engineering.py -i "input_data/COP (Conoco Phillips).csv" 

-fx_feature_engining.py
python fx_feature_engining.py -i "input_data/USD_EUR_daily_full.csv"

-option_feature_engineering.py
python option_feature_engineering.py -i "input_data/options.csv"