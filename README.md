# Wanna get some auton lab server log data?
Here's how:
```
    from main import determineAvailableData, parse_all
    determineAvailableData()
    parse_all(
        pd.read_csv('available_data.csv', parse_dates=['date']),
        excluded_types= set(['daemon'])
    )
```