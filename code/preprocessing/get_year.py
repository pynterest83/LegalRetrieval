import re
from datetime import datetime

"""
catch date in document and check if it is in correct format
@param location: location of date in document
@param mylist: list of word in document
@return True if found date, False if not found date
"""
def check_date(location, mylist):
    if location < 2: return False, ""
    # Check if the date is in the correct format
    check_fisrt = mylist[location-2] + " " + mylist[location-1] + " " + mylist[location]
    content = mylist[location-5] + " " + mylist[location-4] + " " + mylist[location-3] + " " + mylist[location-2] + " " + mylist[location-1] + " " + mylist[location]
    try:
        date_object = datetime.strptime(check_fisrt, "%B %d, %Y")
        return True, check_fisrt, content
    except:
        check_fisrt = mylist[location-1] + " " + mylist[location]
    try:
        date_object = datetime.strptime(check_fisrt, "%B %Y")
        return True, check_fisrt, content
    except:
        check_fisrt = mylist[location]
    try:
        date_object = datetime.strptime(check_fisrt, "%d/%m/%Y")
        return True, check_fisrt, content
    except:
        try:
            date_object = datetime.strptime(check_fisrt, "%m/%d/%Y")
            print(check_fisrt)
            return True, check_fisrt, content
        except:
            return False, ""

'''
get list of year in document

@param doc: document to get year
@return list of year
'''        
def get_list_year(doc):

    _year1 = []
    _full1 = []
    _full11 = []

    mylist = doc.split()

    for i in range(len(mylist)):
        l = mylist[i]
        match = re.match(r'.*([1-2][0-9]{3})', l)
        if match is not None:
            # Then it found a match!
            mylist[i] = match.group(1)
            _check_date = check_date(i, mylist)
            if _check_date[0] == True:
                value = match.group(1)
                if int(value) > 2015: continue
                if int(value) < 1900: continue
                _year1.append(value)
                _full1.append(_check_date[1])
                _full11.append(_check_date[2])

    _year2 = []
    _full2 = []

    mylist = doc.split()

    for i in range(len(mylist)):
        l = mylist[i]
        match = re.match(r'^[([].*([1-2][0-9]{3})', l)
        if match is not None:
            value = match.group(1)
            if int(value) > 2015: continue
            if int(value) < 1900: continue
            _year2.append(value)
            _full2.append(l)

    return _year1, _full1, _year2, _full2, _full11

'''
get biggest year of document

@param doc: document
@return year of document
'''
def get_year(doc):
    max1 = max2 = 0
    _year = get_list_year(doc)
    if len(_year[0]) > 0:
        max1 = max(_year[0])
    if len(_year[2]) > 0:
        max2 = max(_year[2])
    return max(int(max1), int(max2))

'''
concat year into paragraph

@param paragraphs: list of paragraphs
@param year: year of document
@return list of paragraphs with year
'''
def concat_list_year(paragraphs: list, year: int):
    paragraph_year = []
    for paragraph in paragraphs:
        concat_year = {
            "paragraph": paragraph,
            "year": year
        }
        paragraph_year.append(concat_year)

    return paragraph_year