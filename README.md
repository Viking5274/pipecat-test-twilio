## Setup Instructions

Follow these steps to set up and run the project:

### 1. Create a Virtual Environment

First, create a virtual environment using `venv`:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

Activate the virtual environment:

For macOS/Linux:
```bash
source venv/bin/activate
```

For Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the Python Script

Execute the `test_bot.py` script:

```bash
python test_bot.py
```

### 5. Run ngrok

Run ngrok to tunnel HTTP traffic to your local server:

```bash
ngrok http 127.0.0.1:8765
```

## Additional Information

- Make sure `ngrok` is installed and properly configured.
- Update the `requirements.txt` file with all necessary dependencies.
- Adjust the instructions as needed for your specific project requirements.
