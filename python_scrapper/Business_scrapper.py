import time
import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import mysql.connector as sql
from datetime import date
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.wait import WebDriverWait

tdate = "2023-11-13"
category="Gyms"

#https://www.google.com/search?tbs=lf:1,lf_ui:2&tbm=lcl&q=gyms+in+goa&rflfq=1&num=10&sa=X&ved=2ahUKEwiokMrEh67_AhUKXGwGHWdQCwIQjGp6BAgXEAE#rlfi=hd:;si:;mv:[[15.502566499999999,74.0335441],[15.240277400000002,73.8066463]]
driver = webdriver.Chrome(executable_path="E:\\MY_BLOCKS\\python_scrapper\\chromedriver.exe")
driver.get("https://www.google.com/search?q=gyms+in+visakhapatnam&sca_esv=582882630&rlz=1C1CHBF_enIN1042IN1042&biw=1366&bih=611&tbm=lcl&sxsrf=AM9HkKmBTJ2CB6_eu9Fu12rFFvLrsZWSWw%3A1700109005726&ei=zZpVZdX0K6mOxc8P_LCGgAQ&oq=gyms+in&gs_lp=Eg1nd3Mtd2l6LWxvY2FsIgdneW1zIGluKgIIADIEECMYJzIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABEjoH1AAWK8RcAB4AJABAZgBpQGgAakIqgEDMC45uAEDyAEA-AEBwgILEAAYgAQYsQMYgwHCAgcQABiKBRhDwgIKEAAYigUYsQMYQ8ICBxAAGIAEGArCAggQABiKBRiRAsICCxAAGIoFGLEDGJECwgIIEAAYgAQYsQPCAg4QABiKBRixAxjJAxiRAsICCBAAGIoFGJIDwgINEAAYigUYsQMYyQMYQ8ICDRAAGIoFGLEDGIMBGEOIBgE&sclient=gws-wiz-local#rlfi=hd:;si:;mv:[[17.816449,83.3674407],[17.6656673,83.19440480000002]];start:100")
time.sleep(1)
elements = driver.find_elements(By.CSS_SELECTOR, ".vwVdIc")



portalid=135559
memberid=1127
parentportalid=11058

# Get the current date
current_date = datetime.date.today()

# Format the date in the desired format
formatted_date = current_date.strftime("%Y-%m-%d")

mydb=sql.connect(host="61.2.142.91",user="nrktrn_web ",password="nrktrn11",database="nrkindex_trn",port=8306)

mycursor = mydb.cursor()


# Click on each element one by one


for element in elements:
    element.click()
    wait= WebDriverWait(driver,100)
    time.sleep(2)
    titl = ""
    mylink = ""
    addr = ""
    phone = ""

    try:
        titl = driver.find_element(By.XPATH,'//h2[@data-attrid="title"]')
        titl = titl.text
        print(titl)

    except NoSuchElementException:
        pass


    try:
        link = driver.find_element(By.CSS_SELECTOR, "a.dHS6jb")
        mylink=link.get_attribute("href")

        if "google" in mylink:
            mylink = ""

        print(mylink)

    except NoSuchElementException:
        pass

    try:
        addr = driver.find_element(By.XPATH,'//span[@class="LrzXr"]')
        addr=addr.text

        print(addr)
    except NoSuchElementException:
        pass


    try:
        phone_num = driver.find_element(By.XPATH,"//span[contains(@aria-label,'Call')]")
        phone=phone_num.text
        phone=phone.replace(" ", "")
        print(phone)

    except NoSuchElementException:
        pass

    sql = "INSERT INTO kf_vendor(VEND_TITL, PORTAL_ID, MEMBERID,PARENTPORTALID,VEND_CON_ADDR, vend_url, phone,VEND_SDATE,VEND_CATEGRY) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s)"
    val = (titl, portalid, memberid, parentportalid, addr, mylink, phone, tdate,category)
    mycursor.execute(sql, val)
    mydb.commit()




num_loops = 50

for i in range(2, num_loops):
    # Form the dynamic aria-label value based on the loop number
    aria_label_value = f'Page {i}'
    try:
        # Find the link element by its aria-label attribute
        link_element = driver.find_element(By.CSS_SELECTOR, f'a[aria-label="{aria_label_value}"]')

        # Click on the link
        link_element.click()
        time.sleep(10)
        elements = driver.find_elements(By.CSS_SELECTOR, ".vwVdIc")
        for element in elements:
            element.click()
            time.sleep(2)
            titl = ""
            mylink = ""
            addr = ""
            phone = ""

            try:

                titl = driver.find_element(By.XPATH,'//h2[@data-attrid="title"]')
                titl = titl.text
                print(titl)

            except NoSuchElementException:
                pass

            try:
                link = driver.find_element(By.CSS_SELECTOR, "a.dHS6jb")
                mylink = link.get_attribute("href")

                if "google" in mylink:
                    mylink = ""

                print(mylink)

            except NoSuchElementException:
                pass

            try:
                addr = driver.find_element(By.XPATH,'//span[@class="LrzXr"]')
                addr = addr.text

                print(addr)
            except NoSuchElementException:
                pass

            try:
                phone_num = driver.find_element(By.XPATH,"//span[contains(@aria-label,'Call')]")
                phone = phone_num.text
                phone = phone.replace(" ", "")
                print(phone)

            except NoSuchElementException:
                pass

            sql = "INSERT INTO kf_vendor(VEND_TITL, PORTAL_ID, MEMBERID,PARENTPORTALID,VEND_CON_ADDR, vend_url, phone,VEND_SDATE,VEND_CATEGRY) VALUES (%s,%s, %s, %s, %s, %s, %s, %s,%s)"
            val = (titl, portalid, memberid, parentportalid, addr, mylink, phone, tdate, category)
            mycursor.execute(sql, val)
            mydb.commit()

    except NoSuchElementException:
        pass