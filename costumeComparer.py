from costume import Costume
import elementComparer as elemcomp
import attributeComparer as attrcomp
import aggregator as aggre
import enum
from taxonomie import Taxonomie
from attribute import Attribute

""" 
Defines an enum to specify what should be done
if an attribute is missing
"""
class EmptyAttributeAction(enum.Enum):
    ignore = 1
    evaluate_as_zero = 2

"""
Represents the comparer class for costumes
"""
class CostumeComparer:
    def __init__(
        self,
        elementComparer: elemcomp.ElementComparerType = elemcomp.ElementComparerType.wuPalmer,
        attributeComparer: attrcomp.AttributeComparerType = attrcomp.AttributeComparerType.symMaxMean,
        attributeAggregator: aggre.AggregatorType = aggre.AggregatorType.mean,
        missingElementAction: attrcomp.MissingElementAction = attrcomp.MissingElementAction.ignore,
        emptyAttributeAction: EmptyAttributeAction = EmptyAttributeAction.ignore) -> None:

        # Creates the comparer and aggregator due to factory pattern
        self.elementComparer = elemcomp.ElementComparerFactory.create(elementComparer)
        self.attributeComparer = attrcomp.AttributeComparerFactory.create(attributeComparer, self.elementComparer)
        self.attributeAggregator = aggre.AggregatorFactory.create(attributeAggregator)

        self.missingElementAction = missingElementAction
        self.emptyAttributeAction = emptyAttributeAction
        return

    def compare(self, first: Costume, second: Costume) -> float:
        tax = Taxonomie()
        tax.load_all()

        aggregation_values = []

        # Compare color
        aggregation_values.append(
            self.elementComparer.compare(
                tax.get_graph(Attribute.color), 
                first.dominant_color, 
                second.dominant_color))


        # Compare traits
        #aggregation_values.append(
        #    self.attributeComparer.compare(
        #        tax.get_graph(Attribute.traits), 
        #        first.dominant_traits, 
        #        second.dominant_traits))

        # Compare condition
        aggregation_values.append(
            self.elementComparer.compare(
                tax.get_graph(Attribute.condition), 
                first.dominant_condition, 
                second.dominant_condition))

        # Compare traits
        aggregation_values.append(
            self.attributeComparer.compare(
                tax.get_graph(Attribute.stereotype), 
                first.stereotypes, 
                second.stereotypes))

        # Compare condition
        aggregation_values.append(
            self.elementComparer.compare(
                tax.get_graph(Attribute.gender), 
                first.gender, 
                second.gender))

        # Compare condition
        aggregation_values.append(
            self.elementComparer.compare(
                tax.get_graph(Attribute.age_impression), 
                first.dominant_age_impression, 
                second.dominant_age_impression))

        # Compare traits
        aggregation_values.append(
            self.attributeComparer.compare(
                tax.get_graph(Attribute.genre), 
                first.genres, 
                second.genres))

        return self.attributeAggregator.aggregate(aggregation_values)