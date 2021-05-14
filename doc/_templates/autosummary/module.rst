{{ fullname }}
{{ underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
    :toctree: {{ fullname }}
    {% for class in classes %}
    {% if class in members %}
    {{ class }}
    {% endif %}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: {{ fullname }}
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}