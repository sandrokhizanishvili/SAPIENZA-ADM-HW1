## Introduction

# Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else

if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 == 1: #If n is odd
        print('Weird')
    elif n % 2 == 0 and 2<=n<=5: #If n is even and in the inclusive range [2,5]
        print('Not Weird')
    elif n % 2 == 0 and 6<=n<=20: #If n is even and in the inclusive range [6,20]
        print('Weird')
    elif n % 2 == 0 and n > 20: #If n is even and greater than 20 
        print('Not Weird')



# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a+b) # summation
    print(a-b) # substraction
    print(a*b) # multiplication

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a//b) # integer division
    print(a/b) # float division


# Loops

if __name__ == '__main__':
    n = int(input())
    for i in range(n): # iteration over 0....n-1
        print(i**2) # output i^2


# Write a function

def is_leap(year):
    leap = False
    
    # Write your logic here
    if year%4==0 and year%100!=0: # evenly devided by 4 and not 100
        leap = True
    elif year%400==0: # evenly devided by 400
        leap=True
    
    return leap

# Print Function

if __name__ == '__main__':
    n = int(input())
    string = "" # create empty string
    for i in range(1,n+1): # iteration over 1.....n+1 (Ex : 1....5)
        string += str(i) # append values in empty string
    print(string)
    
## Data types

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    coordinates = [[i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n] # iteration over 0...x+1 ; 0...y+1 ; 0...z+1
    print(coordinates)

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr = set(arr) # map fuction returns iterator, converting it to set to remove duplicated elements as well
    arr = sorted(arr, reverse=True) # descending sort
    print(arr[1]) # second place

# Nested Lists

if __name__ == '__main__':
    names = []
    scores = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        names.append(name)
        scores.append(score)
    sec_min = sorted(set(scores))[1] # set - removes duplicates, sorted - sort ascending, [1] - second elemen list (second lowest)
    sec_names = []
    for ind, val in enumerate(scores): # iteration with indices
        if val==sec_min: # identify index where score is second lowest
            sec_names.append(names[ind])
        sec_names.sort()   # sort alphabeticaly   
    
    for i in sec_names:
        print(i)

# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    avg = sum(student_marks[query_name])/len(student_marks[query_name]) # calculating mean
    avg = f"{avg:.2f}" # formating to show 2 numbers after decimal point
    print(avg)

# Lists

if __name__ == '__main__':
    N = int(input())
    arr = [] # Initialize an empty list


    for _ in range(N): # Iterate through each command
        command = input().strip().split()
        cmd_type = command[0]

        if cmd_type == 'insert': # Insert integer e at position i
            i, e = int(command[1]), int(command[2])
            arr.insert(i, e)
        elif cmd_type == 'print': # Print the list
            print(arr)
        elif cmd_type == 'remove': # Delete the first occurrence of integer e
            e = int(command[1])
            arr.remove(e)
        elif cmd_type == 'append': # Insert integer e at the end of the list
            e = int(command[1])
            arr.append(e)
        elif cmd_type == 'sort': # Sort the list ascending
            arr.sort()
        elif cmd_type == 'pop': # Delete the last element from the list
            arr.pop()
        elif cmd_type == 'reverse': # Reverse the list
            arr.reverse()

# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))

## Strings

# sWAP cASE

def swap_case(s):
    swapped = ''.join([char.lower() if char.isupper() else char.upper() for char in s])
    return swapped

# String Split and Join

def split_and_join(line):
    line = line.split(' ') # split string on a whitespace
    line = '-'.join(line) # merge list with '-' symbol
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?

def print_full_name(first, last):
    print(f'Hello {first} {last}! You just delved into python.') # string fromating

# Mutations

def mutate_string(string, position, character):
    string = string[:position] + character + string[position+1:]
    return string

# Find a string

def count_substring(string, sub_string):
    count = 0
    # I'm making sliding window for the string, where each window has the length of sub_string
    for i in range(len(string)-len(sub_string)+1): 
        start_index = i
        end_index = len(sub_string)+i
        
        if string[start_index:end_index] == sub_string:
            count+=1
            
    return count

# String Validators

if __name__ == '__main__':
    s = input()
    alpha_num = 0
    alpha = 0
    digits = 0
    lower = 0 
    upper = 0
    
    for char in s:
        alpha_num += char.isalnum() # is alphanumeric
        alpha += char.isalpha() # is alphabetical 
        digits += char.isdigit() # is digit 
        lower += char.islower() # is lowercase
        upper += char.isupper() # is uppercase
    
    # printing True or False resutls   
    print(alpha_num>0)
    print(alpha>0)
    print(digits>0)
    print(lower>0)
    print(upper>0)

# Text Alignment

thickness = int(input()) 

c = 'H' # The character 'H'

for i in range(thickness): # Top cone
    print((c * (2 * i + 1)).center(thickness * 2))

for i in range(thickness + 1): # Top pillars
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range((thickness + 1) // 2): # Middle belt
    print((c * thickness * 5).center(thickness * 6))

for i in range(thickness + 1): # Bottom pillars
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range(thickness): # Bottom cone
    print((c * (thickness* 2 - 1 - i*2)).rjust(thickness*6-i-1))

# Text Wrap

import textwrap

def wrap(string, max_width):
    wrapped = textwrap.fill(string, max_width)
    return wrapped


# Designer Door Mat

N, M = map(int, input().split())

symbol = '.|.'
welcome = 'WELCOME'
iter_len = N//2

for i in range(iter_len): # first part
    symbols = symbol * (1 + i*2)
    print(symbols.center(M, '-'))


print(welcome.center(M, '-')) # middle part


for i in range(iter_len-1, -1, -1): # bottom part
    symbols = symbol * (1 + i*2)
    print(symbols.center(M, '-'))

# String Formatting

def print_formatted(number):

    width = len(bin(number)) - 2 # to display every elemnt with the maxium width
    for i in range(1,number+1):
        print(f"{i:{width}d} {i:{width}o} {i:{width}X} {i:{width}b}")

# Alphabet Rangoli

def print_rangoli(size):

    # Let's start from the mid phrase, to determine length(width)
    a = ord('a')
    mid_phrase_left = [chr(a+i) for i in range(size-1,-1,-1)]
    mid_phrase_right = [chr(a+i) for i in range(1,size)]
    mid_phrase = mid_phrase_left + mid_phrase_right
    mid_string = '-'.join(mid_phrase)
    width = len(mid_string)
    
    # First upper part (included mid pattern) 
    for i in range(0,size):
        left = [chr(a+i) for i in range(size-1, size-1-i, -1)]
        right = [chr(a+i) for i in range(size-1-i, size)]
        combined = left + right
        combined_string = '-'.join(combined)
        combined_string = combined_string.center(width,'-')
        print(combined_string)
    
    # Second lower part (excluded mid pattern)
    for i in range(size-2,-1,-1):
        left = [chr(a+i) for i in range(size-1, size-1-i, -1)]
        right = [chr(a+i) for i in range(size-1-i, size)]
        combined = left + right
        combined_string = '-'.join(combined)
        combined_string = combined_string.center(width,'-')
        print(combined_string)

# Capitalize!

def solve(s):
    s = s.split(' ')

    fullname = []
    for name in s:
        if len(name) !=0:
            name = name[0].upper() + name[1:]
            fullname  = fullname + [name]
        else:
            fullname  = fullname + [name]

    fullname = ' '.join(fullname)
    
    return fullname

# The Minion Game

def minion_game(string):
    stuart = 0
    kevin = 0
    vowels = "AEIOU"
    
    
    for i in range(len(string)):
        
        score = len(string) - i
        
        # Check if the letter is a vowel or consonant
        if string[i] in vowels:
            kevin += score
        else:
            stuart += score
    
    
    if kevin > stuart:
        print(f'Kevin {kevin}')
    elif stuart > kevin:
        print(f'Stuart {stuart}')
    else:
        print('Draw')

# Merge the Tools!

def merge_the_tools(string, k):
    # your code goes here
    length = len(string) # lenght of string
    n_k = int(length/k) # count of substrings
    for i in range(n_k): # iteration over substrings
        distincts = '' # initialize empty string to add distinct charachters
        substring = string[k*i:k+k*i]
        for char in substring:
            if char not in distincts: # append distinct charachters
                distincts = distincts + char
        print(distincts)


## Sets

# Introduction to Sets

def average(array):
    # your code goes here
    dist_arr = set(array) # distinct elements
    len_dist_arr = len(dist_arr) # lenght of distinct elements
    total = sum(dist_arr) # sum of distinct elements
    avg = total / len_dist_arr # average of distinct elements
    
    return avg

# No Idea!

n, m = map(int, input().split())  
array = list(map(int, input().split())) 
A = set(map(int, input().split()))  
B = set(map(int, input().split()))  

# I tried a lot of methods, all of them failed because of optimization except this one
count = {}

# count occurances of each elemnt in array
for num in array:
    if num in count:
        count[num] += 1
    else:
        count[num] = 1

score = 0

# positive points
for num in A:
    if num in count:
        score += count[num]

# negative points
for num in B:
    if num in count:
        score -= count[num]

print(score)

# Symmetric Difference

M  = int(input())
set_m = set(map(int, input().split()))
N  = int(input())
set_n = set(map(int, input().split()))

diff = set_m.symmetric_difference(set_n) # symmetric difference
diff = sorted(diff) # ascending sort

for i in diff:
    print(i)

# Set .add()

N = int(input())

country_set = set() # initialize empty set

# add one by one countries (duplicated countries won't be added)
for i in range(N):
    country = input()
    country_set.add(country) 

print(len(country_set))

# Set .union() Operation

n = int(input()) # N of English news students
n_set = set(map(int, input().split())) # English news roll numbers
b = int(input()) # N of French news students
b_set = set(map(int, input().split())) # French news roll numbers

union = n_set.union(b_set)

print(len(union))

# Set .intersection() Operation

n = int(input()) # N of English news students
n_set = set(map(int, input().split())) # English news roll numbers
b = int(input()) # N of French news students
b_set = set(map(int, input().split())) # French news roll numbers

union = n_set.intersection(b_set)

print(len(union))

# Set .difference() Operation

n = int(input()) # N of English news students
n_set = set(map(int, input().split())) # English news roll numbers
b = int(input()) # N of French news students
b_set = set(map(int, input().split())) # French news roll numbers

union = n_set.difference(b_set)

print(len(union))

# Set .symmetric_difference() Operation

n = int(input()) # N of English news students
n_set = set(map(int, input().split())) # English news roll numbers
b = int(input()) # N of French news students
b_set = set(map(int, input().split())) # French news roll numbers

union = n_set.symmetric_difference(b_set)

print(len(union))

# Set Mutations

n = int(input())
A = set(map(int, input().split()))
N = int(input())

for _ in range(N):
    operation, _ = input().split()
    
    other_set = set(map(int, input().split()))
    if operation == 'update':
        A.update(other_set)
    elif operation == 'intersection_update':
        A.intersection_update(other_set)
    elif operation == 'difference_update':
        A.difference_update(other_set)
    elif operation == 'symmetric_difference_update':
        A.symmetric_difference_update(other_set)

print(sum(A))

# The Captain's Room

size = int(input())
rooms = list(map(int, input().split()))
rooms_set = set(rooms)

# Calculate sum of all room numbers and sum of unique room numbers
sum_rooms = sum(rooms)
sum_unique_rooms = sum(rooms_set)

# Calculate and print the Captain's room number
captains_room = int((size * sum_unique_rooms - sum_rooms) / (size - 1))
print(captains_room)

# Check Subset

T = int(input())
for _ in range(T):
    n_A = int(input())
    A = set(map(int, input().split()))
    n_B = int(input())
    B = set(map(int, input().split()))
    intersection = A.intersection(B)
    print(len(A) == len(intersection))

# Check Strict Superset

A = set(map(int, input().split()))
n = int(input())
T_F = [] # initialize empty list to append True and False s
for _ in range(n):
    B = set(map(int, input().split()))
    intersection = A.intersection(B)
    T_F.append((len(intersection) == len(B)) and (len(intersection) != len(A))) # Checking logic of superset and addind it in T_F

if False in T_F:
    print(False)
else:
    print(True)


# Set .discard(), .remove() & .pop()

n = int(input()) # the number of elements
n_set = set(map(int, input().split())) # the set
N = int(input()) # the number of commands

# I think because of the random nature of sets and pop, I cannot get the correct answer

for _ in range(N):
    command_inp = input()
    if (command_inp == 'pop') and (len(n_set)>0):
        n_set.pop()
    elif command_inp != 'pop':
        command, value = command_inp.split()
        value = int(value)
        n_set.discard(value)

print(sum(n_set))


## Collections

# collections.Counter()

from collections import Counter

X = int(input())
shoe_sizes = list(map(int, input().split()))
N = int(input())

count = Counter(shoe_sizes) # count each size
income = 0

for _ in range(N):
    size, price = list(map(int, input().split()))
    if count[size] != 0: # check if seller has the option
        income += price 
        count[size] -= 1

print(income)

# DefaultDict Tutorial

from collections import defaultdict

n, m = list(map(int, input().split()))

# preparing defaultdict
d = defaultdict(list)

for i in range(1, n+1):
    A = input()
    d[A].append(i)
    
# print final output
for _ in range(m):
    B = input()
    if len(d[B])==0:
        print('-1')
    else:
        print(' '.join(map(str, d[B])))


# Collections.namedtuple()

from collections import namedtuple

N = int(input())
columns = input().split()

Students = namedtuple('Students', columns)

total_marks=0

for _ in range(N):
    student_data = input().split()
    student = Students(*student_data) # *student upack elemts as seperate arguments
    total_marks += int(student.MARKS)

average_marks = total_marks / N
print(f"{average_marks:.2f}")

# Collections.OrderedDict()

from collections import defaultdict

N = int(input())
d = defaultdict(list)

for _ in range(N):
    data = input().split()
    price = int(data[-1])  # the price is the last element
    item_name = " ".join(data[:-1]) 
    d[item_name].append(price)
    
for key in d:
    print(f"{key} {sum(d[key])}")


# Word Order

from collections import defaultdict
n = int(input())
d = defaultdict(list)

for _ in range(n):
    word = input()
    d[word].append(1)

print(len(d))

count = ''

for i in d:
    count += f"{sum(d[i])} "

print(count)

# Collections.deque()

from collections import deque

N = int(input())
d = deque()

for _ in range(N):
    inp = input()
    if inp == 'pop':
        d.pop()
    elif inp == 'popleft':
        d.popleft()
    else:
        command, value = inp.split()
        value = int(value)
        
        if command == 'append':
            d.append(value)
        else:
            d.appendleft(value)

print(' '.join(map(str, d)))

# Company Logo

from collections import Counter

if __name__ == '__main__':
    s = input()
    count = Counter(s)
    
    sorted_counter = sorted(count.items(), key=lambda x: (-x[1], x[0])) # lambda functions says that order based on -values (which is desceinding), and then based on keys. output is list of tuples 
    
    for i in sorted_counter[:3]:
        print(f"{i[0]} {i[1]}")

# Piling Up!

T = int(input())

for _ in range(T):
    n = int(input())
    blocks = list(map(int, input().split()))
    
    left = 0
    right = n - 1
    max_block = float('inf')
    
    while left <= right:
        # choose the largest one fisrt. If you cannot place it then the asnwer is 'No
        if blocks[left] >= blocks[right]:
            curr_block = blocks[left]
            left += 1
        else:
            curr_block = blocks[right]
            right -= 1
        
        # check whether you can place or not
        if curr_block <= max_block:
            max_block = curr_block
        else:
            print('No')
            break
    else: # this only executes if while fails, that means that you can arrange cubes
        print('Yes')


## Date and Time

# Calendar Module

import calendar

month, day, year = map(int, input().split())

day_of_week = calendar.weekday(year, month, day) # it returns integer [0,6], where 0 means Monday and 6 Sunday

week_days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']

print(week_days[day_of_week])

# Time Delta

import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    # format of date
    fmt = '%a %d %b %Y %H:%M:%S %z'
    
    # convert to timestamo
    dt1 = datetime.strptime(t1, fmt)
    dt2 = datetime.strptime(t2, fmt)
    
    # the absolute difference in seconds
    delta = abs(int((dt1 - dt2).total_seconds()))
    
    return str(delta)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

## Exceptions

# Exceptions
T = int(input())

for _ in range(T):
    a, b = input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print("Error Code:", e)

## Built-ins

# Zipped!

N, X = list(map(int, input().split()))

subjects = []
for _ in range(X):
    subject_scores = list(map(float, input().split()))
    subjects = subjects + [subject_scores]
    
student_scores = list(zip(*subjects))

for i in range(N):
    print(sum(student_scores[i])/len(student_scores[i]))

# Athlete Sort

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    sorted_arr = sorted(arr, key=lambda x: x[k])
    
    for i in sorted_arr:
        print(' '.join(map(str, i)))

# ginortS

S = input()

lower = ''
upper = ''
odd = ''
even = ''

for i in S:
    if i.isalpha():
        if i.islower():
            lower = lower + i
        else:
            upper = upper + i
    else:
        if int(i) % 2 !=0:
            odd = odd + i
        else:
            even = even + i

lower = ''.join(sorted(lower))
upper = ''.join(sorted(upper))
odd = ''.join(sorted(odd))
even = ''.join(sorted(even))


print(lower + upper + odd + even)


## Python Functionals

# Map and Lambda Function

cube = lambda x: x**3# complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    fib = [0,1] # initialize fibonacci list with two elemnts

    for ind in range(2,n):
        fib.append(fib[ind-2] + fib[ind-1])
    
    return fib[:n] # it is necessary to slicing, because it will return incorrect list for n<=1

## Regex and Parsing challenges

# Detect Floating Point Number

import re

T = int(input())

for _ in range(T):
    string = input()
    
    try:
        logic = bool(float(string)+0.0001) & bool(re.search(r"\.", string)) # I'm checking whether float contains dot or not (output is boolean)
        
        print(logic)
    except:
        print(False)

# Re.split()

regex_pattern = r"[,.]"

# Group(), Groups() & Groupdict()

import re

string = input().strip()

# Initialize a variable to store the first repeating character
first_repeating_char = -1

# iterate through the string using range to avoid index error
for i in range(len(string) - 1):
    # Check if the current character is alphanumeric
    if string[i].isalnum():
        # Check for consecutive repetitions
        if string[i] == string[i + 1]:
            first_repeating_char = string[i]
            break

print(first_repeating_char)

# Re.findall() & Re.finditer()

import re

S = input().strip()

# define the regex pattern to match the required substrings
pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])([aeiouAEIOU]{2,})(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'

# find all matches using the regex pattern
matches = re.findall(pattern, S)

if matches:
    for match in matches:
        print(match)
else:
    print(-1)

# Re.start() & Re.end()

import re

string = input()
sub_string = input()

# initialize a list to store the start and end indices
indices = []

# initialize a starting point
start = 0


for i in range(len(string)):
    
    match = re.search(sub_string, string[start:]) # returns None if has not found anything
    
    if match:
        
        start_index = match.start() + start
        end_index = match.end() + start - 1  # -1 to convert to inclusive index
        indices.append((start_index, end_index))
        
        
        start += match.start() + 1  # move to the next character after the current match
    else:
        break  # No more matches


if indices:
    for index in indices:
        print(index)
else:
    print((-1, -1))

# Regex Substitution

import re


N = int(input())

# list to store the modified lines
modified_lines = []

for _ in range(N):
    line = input()
    
    # Replace && with and; || with or
    line = re.sub(r'(?<= )&&(?= )', 'and', line)
    line = re.sub(r'(?<= )\|\|(?= )', 'or', line)
    
    
    modified_lines.append(line)


for modified_line in modified_lines:
    print(modified_line)

# Validating Roman Numerals

regex_pattern = r"^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

# Validating phone numbers

import re

N = int(input())

for _ in range(N):
    number = input().strip()
    
    pattern = r"^[789]\d{9}$"
    
    if re.match(pattern, number):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses

import re
import email.utils

n = int(input())

pattern = r'^[a-zA-Z][a-zA-Z0-9_.-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$'

for _ in range(n):
    name, email_ad = email.utils.parseaddr(input())
    
    if re.match(pattern, email_ad):
        print(email.utils.formataddr((name, email_ad)))

# Hex Color Code

import re

N = int(input())

pattern = r'(?<!^)(#[A-Fa-f0-9]{3,6})\b'

# Loop through each line of CSS code
for _ in range(N):
    line = input()

    # Search for hex colors in each line
    matches = re.findall(pattern, line)
    for match in matches:
        print(match)

# HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attr in attrs:
            # print each attribute and its value (None if no value is given)
            print(f"-> {attr[0]} > {attr[1] if attr[1] else 'None'}")

    
    def handle_endtag(self, tag):
        print(f"End   : {tag}")

    
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attr in attrs:
            # print each attribute and its value (None if no value is given)
            print(f"-> {attr[0]} > {attr[1] if attr[1] else 'None'}")

N = int(input())

# read all lines of HTML
html_code = ""
for _ in range(N):
    html_code += input()

parser = MyHTMLParser()

parser.feed(html_code)

# HTML Parser - Part 2

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        # print multi-line comments
        if '\n' in data:
            print(">>> Multi-line Comment")
            print(data)
        else:
            # print single-line comments
            print(">>> Single-line Comment")
            print(data)

    def handle_data(self, data):
        # print data only if it is not just whitespace
        if data.strip() and data != '\n':
            print(">>> Data")
            print(data)  # leading whitespace
    

  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        # print the tag
        print(tag)
        # print attributes and their values
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")


    def handle_startendtag(self, tag, attrs):
        # print self-closing tags with their attributes
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")


N = int(input())
html = ""
for _ in range(N):
    html += input() + '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating UID

T = int(input())

for _ in range(T):
    uid = input()
    
    # define all boolean logics 
    length = len(uid)==10
    upper_count = sum(1 for char in uid if char.isupper()) >= 2
    digit_count = sum(1 for char in uid if char.isdigit()) >=3
    is_alphanum = uid.isalnum()
    repet = len(set(uid)) == len(uid)
    
    # check logics
    if length & upper_count & digit_count & is_alphanum & repet:
        print('Valid')
    else:
        print('Invalid')

# Validating Credit Card Numbers

T = int(input())

for _ in range(T):
    card_number = input()
    
    # Boolean logics, they are simple, but code looks so complicated to read:D
    
    starts_with_valid_digit = card_number[0] in '456'
    digit_count = sum(1 for char in card_number if char.isdigit()) == 16
    hyphen = all([len(i) in [4, 16] for i in card_number.split('-')])
    
    card_number_nohyphen = card_number.replace('-', '')
    has_no_consecutive_repeats = not any(card_number_nohyphen[i] == card_number_nohyphen[i + 1] == card_number_nohyphen[i + 2] == card_number_nohyphen[i + 3] for i in range(len(card_number_nohyphen) - 3))
    
    if (starts_with_valid_digit and digit_count and hyphen and has_no_consecutive_repeats):
        print('Valid')
    else:
        print('Invalid')

# Validating Postal Codes

regex_integer_in_range = r"^[1-9]\d{5}$"	
regex_alternating_repetitive_digit_pair = r"(?=(\d)(?=\d\1))"

# Matrix Script

import math
import os
import random
import re
import sys


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

# transpose the matrix and decode
decoded_script = ''.join(''.join(matrix[j][i] for j in range(n)) for i in range(m))

# replace non-alphanumeric characters with a space and strip extra spaces
decoded_script = re.sub(r'(?<=\w)([^\w]+)(?=\w)', ' ', decoded_script).strip()

print(decoded_script)

## XML

# XML 1 - Find the Score

def get_attr_number(node):
    # your code goes here
    attr_count = 0
    
    
    for elem in node.iter(): # iteration over every elemnt in tree
        attr_count += len(elem.attrib)  # add the number of attributes for each element
    
    return attr_count

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    
    stack = [(elem, level)] 

    while stack:
        current_elem, current_level = stack.pop()
        current_level += 1  # increment the depth level for the current element
        
        # update maxdepth if the current level is deeper
        if current_level > maxdepth:
            maxdepth = current_level
        
        # add all children of the current element to the stack
        for child in current_elem:
            stack.append((child, current_level))

## Closures and Decorations

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        standardized_numbers = ["+91 {} {}".format(num[-10:-5], num[-5:]) for num in l]
        return f(standardized_numbers)
    return fun

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        
        # sort people by age (index 2)
        sorted_people = sorted(people, key=lambda x: int(x[2]))
        
        # apply the name format function to each person in the sorted list
        return [f(person) for person in sorted_people]
    return inner

## Numpy

# Arrays

import numpy

def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array(arr, dtype=float)[::-1]

# Shape and Reshape

import numpy

arr = list(map(int, input().split()))

print(numpy.array(arr).reshape(3, 3))

# Transpose and Flatten

import numpy


N, M = list(map(int, input().split()))

arr = []
for _ in range(N):
    arr = arr + list(map(int, input().split()))
    
print(numpy.transpose(numpy.array(arr).reshape(N, M)))
print(numpy.array(arr))


# Concatenate

import numpy



N, M, P = list(map(int, input().split()))

array_1 = numpy.array([input().strip().split() for _ in range(N)], dtype=int)
array_2 = numpy.array([input().strip().split() for _ in range(M)], dtype=int)

result = numpy.concatenate((array_1, array_2), axis=0)

print(result)
    
# Zeros and Ones

import numpy



shape = list(map(int, input().split()))

print(numpy.zeros(shape, dtype=int))
print(numpy.ones(shape, dtype=int))


# Eye and Identity

import numpy


N, M = list(map(int, input().split()))

numpy.set_printoptions(legacy='1.13')

print(numpy.eye(N, M, k=0))


# Array Mathematics

import numpy

N, M = map(int, input().split())

A = numpy.array([list(map(int, input().split())) for _ in range(N)])

B = numpy.array([list(map(int, input().split())) for _ in range(N)])

print(A + B) # Addition
print(A - B) # Subtraction
print(A * B) # Multiplication
print(A // B) # Integer Division
print(A % B) # Modulus
print(A ** B) # Power

# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')


array = numpy.array(list(map(float, input().split())))

print(numpy.floor(array))
print(numpy.ceil(array))
print(numpy.rint(array))

# Sum and Prod

import numpy



N, M = list(map(int, input().split()))

array = numpy.array([list(map(int, input().split())) for _ in range(N)])

np_sum = numpy.sum(array, axis=0) # sum
np_prod = numpy.product(np_sum) # product

print(np_prod)

# Min and Max

import numpy


N, M = list(map(int, input().split()))

array = numpy.array([list(map(int, input().split())) for _ in range(N)])

np_min = numpy.min(array, axis=1) # min
np_max = numpy.max(np_min) # max

print(np_max)

# Mean, Var, and Std

import numpy

N, M = list(map(int, input().split()))

array = numpy.array([list(map(int, input().split())) for _ in range(N)])


print(numpy.mean(array, axis=1)) # mean
print(numpy.var(array, axis=0)) # variance
print(round(numpy.std(array),11)) # standart deviation, rounded to get the same answer

# Dot and Cross

import numpy



N = int(input())

A = numpy.array([list(map(int, input().split())) for _ in range(N)])
B = numpy.array([list(map(int, input().split())) for _ in range(N)])

print(numpy.dot(A, B)) 

# Inner and Outer

import numpy



A = numpy.array(list(map(int, input().split())))
B = numpy.array(list(map(int, input().split())))

print(numpy.inner(A, B))
print(numpy.outer(A, B))

# Polynomials

import numpy



P = list(map(float, input().split()))
x = int(input())


print(numpy.polyval(P, x))

# Linear Algebra

import numpy


N = int(input())

A = [list(map(float, input().split())) for _ in range(N)]

print(round(numpy.linalg.det(A),2))

# Birthday Cake Candles

import math
import os
import random
import re
import sys


def birthdayCakeCandles(candles):
    # Write your code here
    tallest = max(candles) # get tallest height
    
    tallest_tf = [i==tallest for i in candles] # true, false logic

    return sum(tallest_tf) # sum will return count of tallest candles 
        
    
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# Number Line Jumps

import math
import os
import random
import re
import sys


def kangaroo(x1, v1, x2, v2):
    # Case when velocities are the same
    if x1 == x2:

        return 'YES'
    
    # check if the difference in positions is divisible by the difference in velocities
    # and ensure the faster kangaroo starts behind
    if v1 > v2 and (x2 - x1) % (v1 - v2) == 0:
        return 'YES'
    
    return 'NO'
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising

import math
import os
import random
import re
import sys


def viralAdvertising(n):
    # Write your code here
    shared = 5
    liked = 2
    cum_liked = 2
    
    for _ in range(1, n):
        
        shared = liked*3
        liked = shared//2
        cum_liked += liked
    
    return cum_liked
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    # Write your code here
    n_sum = sum(int(digit) for digit in n.strip())
    
    total_sum = n_sum * k
    
    def compute_super_digit(num):
        if num < 10: # contains only one digit
            return num # the code finishes there
        else: # otherwise we need sumation
            return compute_super_digit(sum(int(digit) for digit in str(num)))
            
    return compute_super_digit(total_sum) # recursion, it finishes when there will be only one digit


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1

import math
import os
import random
import re
import sys


def insertionSort1(n, arr):
    # Write your code here
    value = arr[n-1]

    for i in range(2, n+2):
        if value < arr[n-i] and i!=n+1: # check if previous element is greater or not
            arr[n+1-i] = arr[n-i] # if yes then put previous element on the current index
            print(' '.join(map(str, arr)))
        elif i==n+1: # if value is the smallest then it should go on the first index
            arr[0] = value
            print(' '.join(map(str, arr)))
            break
        else:
            arr[n-i+1] = value
            print(' '.join(map(str, arr)))
            break

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2

import math
import os
import random
import re
import sys




def insertionSort2(n, arr):
    
    # start from first element and sort with respect to left side
    for i in range(1, n):
        value = arr[i] # set value of ith element
        j = i - 1
        
        # if element is greater than value shift by one index 
        while j>=0 and arr[j] > value:
            arr[j + 1] = arr[j]
            j -= 1
        
        # place value in the correct position  
        arr[j + 1] = value
        
        print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)


