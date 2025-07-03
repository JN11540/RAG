import pandas as pd

class WeddingTableArranger:
    def __init__(self, sheet_url):
        self.sheet_url = sheet_url

    def _get_sheet_data(self):
        """
        Reads the Google Sheet content from the provided URL.
        """
        try:
            # The URL needs to be modified for direct CSV export
            csv_url = self.sheet_url.replace("/edit?resourcekey=", "/gviz/tq?tqx=out:csv&gid=")
            csv_url = csv_url.split('#gid=')[0] + '&gid=' + self.sheet_url.split('#gid=')[1]
            df = pd.read_csv(csv_url)
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading Google Sheet: {e}")
            return []

    def arrange_table_numbers(self):
        """
        Reads the spreadsheet content, adds a 'table_num' column,
        and assigns table numbers based on attendance and relationship.
        """
        csv_json = self._get_sheet_data()
        if not csv_json:
            return []

        for row in csv_json:
            contact_number = row.get("Your contact number")
            can_attend = row.get("Can you attend the wedding?")
            relationship = row.get("What is your relationship with the newcomer?")

            # Call method2 to get the table number
            table_num = self._assign_table_number(can_attend, relationship)
            row["table_num"] = table_num

        return csv_json

    def _assign_table_number(self, can_attend, relationship):
        """
        Assigns a table number based on attendance and relationship.
        """
        if can_attend == "I want to participate! Witness the happy moment together":
            if relationship == "Groom's relatives":
                table_num = 1
            elif relationship == "Groom's friend":
                table_num = 2
            elif relationship == "Groom's colleague":
                table_num = 3
            elif relationship == "Bride's relatives":
                table_num = 4
            elif relationship == "Bride's friend":
                table_num = 5
            elif relationship == "Bride's colleague":
                table_num = 6
            else:
                table_num = -1  # Default for participating but unknown relationship
        else:
            table_num = -1  # Not attending
        return table_num

# --- How to use the class ---
# Your Google Sheet URL
sheet_url = "https://docs.google.com/spreadsheets/d/1Jz-iZmZ6JLjGzV5xR75lX1k8VPfoZXAlY5TLWZmnMWY/edit?resourcekey=&gid=1377610396#gid=1377610396"

# Create an instance of the class
arranger = WeddingTableArranger(sheet_url)

# Get the data with assigned table numbers
result_data = arranger.arrange_table_numbers()

# Print the results (optional)
if result_data:
    for row in result_data:
        print(row)
else:
    print("No data processed or an error occurred.")