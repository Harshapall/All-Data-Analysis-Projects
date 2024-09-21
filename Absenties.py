import csv


def take_attendance():
    student_name = input("Enter student name: ")
    present = input("Is the student present? (yes/no): ").lower()

    if present == 'yes':
        status = 'Present'
    else:
        status = 'Absent'

    # Write the attendance record to a CSV file
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_name, status])


def view_attendance():
    # Read and display attendance records from the CSV file
    with open('attendance.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(f"Name: {row[0]}, Status: {row[1]}")


# Main loop
while True:
    print("\n1. Take Attendance")
    print("2. View Attendance")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        take_attendance()
    elif choice == '2':
        view_attendance()
    elif choice == '3':
        break
    else:
        print("Invalid choice. Please try again.")
