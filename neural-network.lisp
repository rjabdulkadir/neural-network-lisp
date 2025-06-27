;;;; Lisp transaltion of...
;;;; Source code to go with "Introduction to implementing neural networks"
;;;; Copyright 2015, Arjan van de Ven
;;;; Provided under the terms of the Apache-2.0 license (see LICENSE file)

(defparameter *synapses* 3)

(defparameter *inputs* 2)
(defparameter *outputs* 1)
(defparameter *hidden* 2)
(defparameter *bias* 1)
(defparameter *netsize* (+ *inputs* *outputs* *hidden* *bias*))

;;; convenience constants
(defparameter *hidden-start* (+ *bias* *inputs*))
(defparameter *hidden-end* (+ *bias* *inputs* *hidden*))
(defparameter *output-start* (+ *bias* *inputs* *hidden*))
(defparameter *output-end* (+ *bias* *inputs* *hidden* *outputs*))

(defparameter *maxinput* 4)

;;; a network input is defined as a class
(defclass input () (values output))

;;; inputs defined as an array of input type instances
(defparameter *all-inputs* (make-array *maxinput*))
(dotimes (i *maxinput*)
	(setf (elt *all-inputs* i) (make-instance 'input)))

;;; Filling in the training set. 
;;; For simple networks one can hardcode the inputs like this,
;;; but for more complex problems it's usually more convenient
;;; to load these from a file on disk.
(defun create-training-set ()
  ;; values of the inputs are defined as a list
  (setf (slot-value (elt *all-inputs* 0) 'values) '(0.0 0.0))
  (setf (slot-value (elt *all-inputs* 0) 'output) 0)
  (setf (slot-value (elt *all-inputs* 1) 'values) '(1.0 0.0))
  (setf (slot-value (elt *all-inputs* 1) 'output) 1)
  (setf (slot-value (elt *all-inputs* 2) 'values) '(0.0 1.0))
  (setf (slot-value (elt *all-inputs* 2) 'output) 1)
  (setf (slot-value (elt *all-inputs* 3) 'values) '(1.0 1.0))
  (setf (slot-value (elt *all-inputs* 3) 'output) 0))

;;; The basic neuron defined as a class
(defclass neuron ()
  ;; inputs and weights are defined as arrays
  ((value :initform 0.0)
   (inputs :initform (make-array 3 :initial-element 0))
   (weights :initform (make-array 3 :initial-element 0.0))))

;;; A network is a set of neurons
;;;
;;; For convenience this is represented as a flat array, where
;;; the neurons are, in order,
;;;
;;; count	type
;;; =========================================================================
;;; BIAS	The bias neuron(s) for getting fixed values into the network
;;; INPUTS	The inputs to the network
;;; HIDDEN	The hidden layer neurons
;;; OUTPUTS	The outputs of the network

;;; The network is implemented as an array of neuron class instances
(defparameter *network* (make-array *netsize* :element-type 'neuron))
(dotimes (i *netsize*)
	(setf (elt *network* i) (make-instance 'neuron)))

(defparameter *learning-rate* 0.5)

;;; The "activation function" of a neuron
(defun activation (x)
	(/ 1 (+ 1 (exp (- x)))))

;;; caluclates the output of one neuron for a set of inputs
(defun calculate-neuron (index)
  (let ((neuron (elt *network* index))
        (input-sum 0.0))
    (setf input-sum (loop for i from 0 below *synapses*
                          sum (* (elt (slot-value neuron 'weights) i)
                                 (slot-value (elt *network* (elt (slot-value neuron 'inputs) i)) 'value))))
    (format t "input-sum: ~d ~%" input-sum)
    (setf (slot-value (elt *network* index) 'value) (activation input-sum))))

;;; calculate the output of the whole network
(defun calculate-network ()
  (loop for i from (+ *inputs* 1) below *netsize* do (calculate-neuron i)))

;;; changes the weight of an input from a delta
(defun update-weight (neuron index delta)
  (setf (elt (slot-value neuron 'weights) index) (+ (elt (slot-value neuron 'weights) index) delta)))

;;; computes the outputs of the neurons for one set of inputs,
;;; amd adjust the weights accordingly
(defun compute-one-input (input)
  (let ((error-sum 0.0))
    ;; set the BIAS neuron to 1
    (setf (slot-value (elt *network* 0) 'value) 1.0)
    ;; set the input neurons to the values of the training data
    (dotimes (i *inputs*)
      (setf (slot-value (elt *network* (+ *bias* i)) 'value)
            (nth i (slot-value input 'values))))
    (calculate-network)
    ;; now calculate the error for all outputs
    (dotimes (i *outputs*)
      (let* ((neuron (elt *network* (+ *output-start* i) ))
             (error-output (- (slot-value input 'output) (slot-value neuron 'value)))
             ;; back propagation: the error propagates back via the inputs of the output neuron
             (gradient (* (slot-value neuron 'value) (- 1.0 (slot-value neuron 'value)) error-output)))
        (setf error-sum (+ error-sum (expt error-output 2)))
        (format t "error-sum!!! ~d" error-sum)
        (dotimes (h (+ *bias* *hidden*))
          (let* ((hidden-neuron (elt *network* (elt (slot-value neuron 'inputs) h)))
                 ;; calculate the delta
                 (delta (* *learning-rate* gradient (slot-value hidden-neuron 'value)))
                 ;; calculate the (recursive) gradient
                 (hidden-gradient (* (slot-value hidden-neuron 'value) (- 1 (slot-value hidden-neuron 'value))
                                     gradient (elt (slot-value neuron 'weights) h))))
            ;; adjust the weight
            (update-weight neuron h delta)
            ;; further propagate the error to the input layer
            (dotimes (inp (+ *bias* *inputs*))
              (let* ((input-neuron (elt *network* (elt (slot-value hidden-neuron 'inputs) inp)))
                     (delta (* *learning-rate* (slot-value input-neuron 'value) hidden-gradient)))
                (update-weight hidden-neuron inp delta))
                                        ;(format t "~d ~d ~d ~d ~%" i h inp error-sum)
              )))))
    error-sum))

;;; runs above for all inputs
(defun compute-for-all-inputs ()
  (let ((error-sum 0.0))
    (dotimes (i *maxinput*)
      (setf error-sum (+ error-sum (compute-one-input (elt *all-inputs* i)))))
    (format t "error-sum (cfai) ~d ~%" error-sum)
    error-sum))

;;; sets up the interconnections between neurons
(defun allocate-network ()
  (dotimes (i *netsize*)
    (format t "~d ~d ~d ~%" i (slot-value (elt *network* i) 'weights) (slot-value (elt *network* i) 'inputs)))
  ;; create the network structure
  ;; the hidden layer neurons connect from all input and bias neurons, with random weight
  (loop for i from *hidden-start* below *hidden-end* do
    (dotimes (j (+ *bias* *inputs*))
      (setf (elt (slot-value (elt *network* i) 'inputs) j) j)
      (setf (elt (slot-value (elt *network* i) 'weights) j) (- (random 2.0) 1)))
    (format t "1st loop: ~d ~d ~d ~d ~%" i (elt *network* i) (slot-value (elt *network* i) 'weights) (slot-value (elt *network* i) 'inputs)))
  ;; the output neurons connect from all hidden neurons, with random weight
  (loop for i from *output-start* below *output-end* do
    ;; setup the link to the bias neuron
    (setf (elt (slot-value (elt *network* i) 'inputs) 0) 0)
    (setf (elt (slot-value (elt *network* i) 'weights) 0) (- (random 2.0) 1))
    ;; connect up each of the neurons to the input layer
    (dotimes (j *hidden*)
      (setf (elt (slot-value (elt *network* i) 'inputs) (+ j *bias*)) (+ *hidden-start* j))
      (setf (elt (slot-value (elt *network* i) 'weights) (+ j *bias*)) (- (random 2.0) 1)))
    (format t "2nd loop: ~d ~d ~d ~d ~%" i (elt *network* i) (slot-value (elt *network* i) 'weights) (slot-value (elt *network* i) 'inputs)))
  (dotimes (i *netsize*)
    (format t "~d ~d ~d ~d ~%" i (elt *network* i) (slot-value (elt *network* i) 'weights) (slot-value (elt *network* i) 'inputs))))

;;; create a dot file showing network and weights
(defun output-network (file-name)
  (with-open-file (stream file-name :direction :output)
    (format stream "digraph G {~%")
    (format stream "~C { ~%" #\tab)
    (format stream "~C~C rank=source;~%" #\tab #\tab)
    (dotimes (i (+ *inputs* *bias*))
      (format stream "~C~C~d;~%" #\tab #\tab i))
    (format stream "~C}~%" #\tab)
    (format stream "~C { ~%" #\tab)
    (format stream "~C~C rank=sink;~%" #\tab #\tab)
    (loop for i from *output-start* below *output-end* do
          (format stream "~C~C~d;~%" #\tab #\tab i))
    (format stream "~C}~%" #\tab)
    (loop for i from (+ *inputs* *bias*) below *netsize* do
          (dotimes (j *synapses*)
            (if (> (abs (elt (slot-value (elt *network* i) 'weights) j)) 0.001)
                (format stream "~C~d -> ~d [label = ~$] ; ~%" #\tab (elt (slot-value (elt *network* i) 'inputs) j) i (elt (slot-value (elt *network* i) 'weights) j)))))
    (format stream "~Csubgraph outputcluster { ~%" #\tab)
    (dotimes (i (+ *inputs* *bias*))
          (format stream "~C~C~d;~%" #\tab #\tab i))
    (loop for i from (- *netsize* *outputs*) below *netsize* do
          (format stream "~C~C~d;~%" #\tab #\tab i))
    (format stream "~C}~%" #\tab)
    (format stream "}~%")))

;;; evaluate the result of one input
(defun eval-network (input)
  (setf (slot-value (elt *network* 0) 'value) 1.0)
  (dotimes (i *inputs*)
    (setf (slot-value (elt *network* (+ *bias* i)) 'value) (elt (slot-value input 'values) i)))
  (calculate-network)
  (format t "Output is: ~d ~%" (slot-value (elt *network* 5) 'value)))

;;; main program
(defun main ()
  (create-training-set)
  (allocate-network)
  (let ((error-sum (compute-for-all-inputs)))
    (format t "Error is ~d ~%" error-sum)
    (dotimes (i 5000)
      (setf error-sum (compute-for-all-inputs))
      (format t "Round ~d Error is ~d ~%" i error-sum))))

