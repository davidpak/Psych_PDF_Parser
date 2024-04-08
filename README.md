# Psychology PDF Parser
This script will allow you to input PDFs containing 
information about Psychology tests and surveys and will output
the necessary metadata into a CSV.

## Layout
* `.pytest_cache` - IGNORE
* `flagged` - IGNORE
* `out` - Directory containing the CSV file the script will write to
* `src` - Directory containing the script that will be run
* `test_files` - Directory where all input PDF files will live

## Setup

### Setting up spreadsheet
Navigate to the `out` directory. There should be a file named
`output.csv`, this is the CSV file that the script will write to after generating the proper metadata.
If you wish for the script to write to an existing spreadsheet in your possession, 
drag and drop your desired spreadsheet into the `out` directory and rename the spreadhseet to `output.csv`. If the IDE asks if you want
to replace the current file under that name, click yes.

***NOTE***: The script will not work for any other file type besides `.csv` so please convert any
`.xlsx`, `.xlsm`, or other files to `.csv` format before proceeding.

### Installing all dependencies
After that has been properly setup, open a new terminal.
This can be done using the following keyboard shortcut:

Windows: ``ctrl + shift + ` `` \
Mac : ``ctrl + ` ``

Or by clicking on the `View` tab and selecting `Terminal`

Once you are in the terminal, run the following command (you can just copy and paste this into the terminal and press enter):

`pip install -r requirements.txt`

This will install all necessary dependencies in order for the application to work.

### Setting up the OpenAI API key
The application requires an OpenAI API key, if you do not have an OpenAI API key, please activate one by following the 
instructions listed here:

https://platform.openai.com/docs/quickstart?context=python

Once the API key has been activated, create a new file in the root directory of the project and name it `.env`.
Within the `.env` file write the following line:

`OPENAI_API_KEY=<YOUR_API_KEY>`

## Using the Application
In order to use the script, open a new terminal in your IDE. This can be done using the following keyboard shortcut:

**Windows**: ``ctrl + shift + ` `` \
**Mac** : ``ctrl + ` ``

Or by clicking on the `View` tab and selecting `Terminal`

From the terminal, navigate to the `src` directory by running the following command:

`cd src`

Finally, to run the application, run the following command:

`python main.py ../test_files/<name_of_your_pdf>.pdf`

After around 30 seconds, the console should produce the following message:

`Cleaned response has been saved to: ../out/output.csv
`

Indicating that the metadata has been written into `output.csv`
