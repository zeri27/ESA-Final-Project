# ESA-Final-Project
Final Project for DSAIT4020 Course TU Delft

## Data format

Data files are included in the `data` folder. The file names follow the format:

> FloorNumber_RoomName_MeterName_MeasuredData_Unit

The data is structured in `csv` files with two columns:

- `Time`: datetimestamp of the data
- `Value`: data value (unit described in title)

Some concerns:

- Data is not guaranteed to be complete
- Different data measurements do not necessarily overlap in time with other measurements
- There may be measurement errors
