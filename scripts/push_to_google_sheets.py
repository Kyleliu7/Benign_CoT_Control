import os
import openpyxl
import gspread
import google.auth
from pathlib import Path

SPREADSHEET_ID = "1kHkib3oNXufeh7oN1vt3nJKr9iz0SOQT9KnZcjMLCT8"
EXCEL_PATH = Path(r"C:\Users\bryan\OneDrive\Desktop\CoT Distill\results\ALL_EXPERIMENTS_MASTER_SPREADSHEET.xlsx")

def sync_to_google_sheets():
    print(f"Reading Excel workbook from: {EXCEL_PATH}")
    wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
    
    print("Authenticating with Google Sheets...")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    try:
        credentials, project = google.auth.default(scopes=scopes)
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(SPREADSHEET_ID)
        print(f"Connected to Google Spreadsheet: '{sh.title}' (ID: {SPREADSHEET_ID})")
    except Exception as e:
        print(f"\n[!] Authentication Error: {e}")
        print("To authorize access, please run the following command in PowerShell / Terminal:")
        print('  gcloud auth application-default login --scopes="https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive"')
        print("Then rerun this script.\n")
        return False

    existing_worksheets = {ws.title: ws for ws in sh.worksheets()}
    
    for sheetname in wb.sheetnames:
        excel_ws = wb[sheetname]
        data = []
        for row in excel_ws.iter_rows(values_only=True):
            # Clean none values
            row_clean = ["" if v is None else v for v in row]
            if any(row_clean): # skip completely empty rows
                data.append(row_clean)
        
        if not data:
            continue
            
        print(f"Syncing sheet: '{sheetname}' ({len(data)} rows, {len(data[0])} cols)...")
        if sheetname in existing_worksheets:
            gs_ws = existing_worksheets[sheetname]
            gs_ws.clear()
        else:
            gs_ws = sh.add_worksheet(title=sheetname, rows=len(data) + 10, cols=len(data[0]) + 5)
            existing_worksheets[sheetname] = gs_ws
            
        gs_ws.update(range_name="A1", values=data)
        
    # Delete default 'Sheet1' if it was replaced and empty
    if "Sheet1" in existing_worksheets and "Sheet1" not in wb.sheetnames:
        try:
            sh.del_worksheet(existing_worksheets["Sheet1"])
        except Exception:
            pass
            
    print("\nAll sheets successfully pushed and updated in Google Sheets!")
    print(f"Spreadsheet URL: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit")
    return True

if __name__ == "__main__":
    sync_to_google_sheets()
