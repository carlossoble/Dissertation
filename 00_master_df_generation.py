import yfinance as yf
import numpy as np
import pandas as pd
import iisignature
np.set_printoptions(suppress=True)


start_date = '2003-12-31'
end_date = '2024-12-31'
window_size = 50
points_ticker = 30
points_ref = 5
signature_degree = 3
ibex35_tickers = ["ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC", "BBVA.MC", "CABK.MC", "CLNX.MC",
                  "COL.MC", "ELE.MC", "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC", "IDR.MC",
                  "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC", "PUIG.MC", "RED.MC", "REP.MC",
                  "ROVI.MC", "SAB.MC", "SAN.MC", "SCYR.MC", "SLR.MC", "TEF.MC", "UNI.MC"]
bankinter = "BKT.MC"  # Distorted data before mid 2005
start_date_bankinter='2005-05-31'
ref_tickers = ['SAN.MC', 'IBE.MC', 'ITX.MC']



ref_data = yf.download(ref_tickers, start=start_date, end=end_date)['Close'].pct_change().dropna()*100
master_df=pd.DataFrame()
for ticker in ibex35_tickers:
    print(ticker)
    ticker_data = yf.download(ticker, start=start_date, end=end_date)['Close'].pct_change().dropna()*100
    common_dates = ref_data.index.intersection(ticker_data.index)
    ref_data_ticker = ref_data.loc[common_dates]
    ticker_data = ticker_data.loc[common_dates]
    data_signature=np.column_stack([ticker_data.to_numpy(), ref_data_ticker.to_numpy()])
    rows=[]
    for n in range(window_size, len(common_dates)):
        path = data_signature[n-window_size:n]
        signature=iisignature.sig(path, signature_degree)
        levy_area=np.array([signature[5]-signature[8], signature[6]-signature[12], signature[7]-signature[16]])
        signature_filtered=signature[np.r_[0:7,17:32]]
        row = [ticker, 
           common_dates[n]] + \
          ticker_data.iloc[n-points_ticker:n].values.flatten().tolist() + \
          ref_data['SAN.MC'].iloc[n-points_ref:n].values.flatten().tolist() + \
          ref_data['IBE.MC'].iloc[n-points_ref:n].values.flatten().tolist() + \
          ref_data['ITX.MC'].iloc[n-points_ref:n].values.flatten().tolist() + \
          signature_filtered.tolist() + \
          levy_area.tolist() + \
          ticker_data.iloc[n].values.flatten().tolist()
        rows.append(row)
    iteration_df=pd.DataFrame(rows)
    master_df=pd.concat([master_df, iteration_df], ignore_index=True)
    
print(bankinter)
ticker_data=yf.download(bankinter, start=start_date_bankinter, end=end_date)['Close'].pct_change().dropna()*100
common_dates = ref_data.index.intersection(ticker_data.index)
ref_data_ticker = ref_data.loc[common_dates]
ticker_data = ticker_data.loc[common_dates]
data_signature=np.column_stack([ticker_data.to_numpy(), ref_data_ticker.to_numpy()])
rows=[]
for n in range(window_size, len(common_dates)):
    path = data_signature[n-window_size:n]
    signature=iisignature.sig(path, signature_degree)
    levy_area=np.array([signature[5]-signature[8], signature[6]-signature[12], signature[7]-signature[16]])
    signature_filtered=signature[np.r_[0:7,17:32]]
    row = [bankinter, 
       common_dates[n]] + \
      ticker_data.iloc[n-points_ticker:n].values.flatten().tolist() + \
      ref_data['SAN.MC'].iloc[n-points_ref:n].values.flatten().tolist() + \
      ref_data['IBE.MC'].iloc[n-points_ref:n].values.flatten().tolist() + \
      ref_data['ITX.MC'].iloc[n-points_ref:n].values.flatten().tolist() + \
      signature_filtered.tolist() + \
      levy_area.tolist() + \
      ticker_data.iloc[n].values.flatten().tolist()
    rows.append(row)
iteration_df=pd.DataFrame(rows)
master_df=pd.concat([master_df, iteration_df], ignore_index=True)


columns = (
        ['Ticker', 'Date'] +
        [f'ticker_data_pre{points_ticker-i}' for i in range(points_ticker)] +
        [f'SAN_pre{points_ref-i}' for i in range(points_ref)] +
        [f'IBE_pre{points_ref-i}' for i in range(points_ref)] +
        [f'ITX_pre{points_ref-i}' for i in range(points_ref)] +
        [f'signature_{i}' for i in range(len(signature_filtered))] +
        [f'levy_area_1{i}' for i in range(2,5)] +
        [f'ticker_data_target']
    )
master_df.columns=columns


master_df.to_csv('master_df.csv', index=False)
master_df.head(5).to_csv('master_df_sample.csv', index=False)
