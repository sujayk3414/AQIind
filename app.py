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
