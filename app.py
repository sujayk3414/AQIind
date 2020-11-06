{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [05/Nov/2020 12:01:18] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [05/Nov/2020 12:01:29] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [05/Nov/2020 13:10:26] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "import pickle\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = pickle.load(open('model.pkl', 'rb'))\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('model.html')\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    '''\n",
    "    For rendering results on HTML GUI\n",
    "    '''\n",
    "    int_features = [int(x) for x in request.form.values()]\n",
    "    final_features = [np.array(int_features)]\n",
    "    prediction = model.predict(final_features)\n",
    "\n",
    "    output = round(prediction[0], 2)\n",
    "\n",
    "    return render_template('model.html', prediction_text=output)\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
