If not using the built-in ``load_tools`` and ``process_tools``, one can directly import the ``se3_class`` and make sure the inputs match the types specified in ``se3_class.py``



Note: 

``p_in`` is position input

``q_in`` is orientation input (scipy rotation object)

``p_out`` is velocity 

``q_out`` is next expected orientation (scipy rotation object)

Their shapes and elements change over the "processing". Refer to ``se3_class.py`` for the *FINAL* type and size 


enable adjust_cov given a single trajectory; disable when the data provided is sufficiently large