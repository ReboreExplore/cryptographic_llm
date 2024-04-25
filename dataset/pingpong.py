import csv

def parse_csv(input_file, output_file):
    with open(input_file, 'r') as csv_file:
        reader = csv.reader(csv_file, quotechar='"', delimiter=',',
                            skipinitialspace=True)
        data = list(reader)
    
    for row in data:
        print(row)

    # Function to add quotes to a field if it doesn't have quotes already
    def add_quotes(field):
        if not (field.startswith('"') and field.endswith('"')):
            return f'"{field}"'
        else:
            return field

    # Modify data
    for row_index, row in enumerate(data):
        for col_index, field in enumerate(row):
            data[row_index][col_index] = add_quotes(field)

    # Write modified data to a new CSV file
    with open(output_file, 'w', newline='') as csv_output_file:
        writer = csv.writer(csv_output_file)
        writer.writerows(data)

if __name__ == "__main__":
    input_file = "test_dataset_numbertheory.csv"
    output_file = "test_dataset_numbertheory_out.csv"

    parse_csv(input_file, output_file)
    print("CSV parsing complete!")
