if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd

@transformer
def transform(tables: dict, *args, **kwargs):
    
    #eliminacion de duplicados en todas las tablas
    for name in tables:
        tables[name] = pd.DataFrame(tables[name])
        tables[name].drop_duplicates(inplace=True)

    print(tables["orders"])
    
    #eliminacion de filas que tienen el valor nulo en add_to_cart_order
    tables["orders"]=tables["orders"][tables["orders"]["add_to_cart_order"].notna()]

            #Convertir a 0 los valores de days_since_prior_order en 'INSTACART'

    tables["instacart"].loc[tables["instacart"]["order_number"]<2,"days_since_prior_order"]=0

    #Convertir en int order number
    tables["instacart"]["order_number"]=tables["instacart"]["order_number"].astype(int)

    #votar price de la tabla products ya que no hay valores
    tables["products"].drop(columns=["price"], inplace=True)

    #Convertir a NaN verdadero para posterirmente reliminar todos los valores de NAN
    tables["products"]["product_name"].replace(["", "NaN"], pd.NA, inplace=True)
    tables["products"]=tables["products"].dropna()

    #Realizar el STAR ESQUEMA

    factOrder=tables["orders"]
    factInstaCart=tables["instacart"]

    productsDim=tables["products"]

    productsDim.drop(columns=["aisle_id","department_id"], inplace=True)

    aislesNames=tables["aisles"]["aisle"]
    productsDim["aisle_name"]=aislesNames

    departmentNames=tables["departments"]["department"]
    productsDim["department_name"]=departmentNames

    return {

        "fact_order":factOrder,
        "fact_instacart":factInstaCart,
        "productsDim":productsDim

    }


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
