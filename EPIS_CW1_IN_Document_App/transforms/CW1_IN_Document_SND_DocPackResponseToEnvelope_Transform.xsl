```xsl
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema"
                xmlns:fn="http://www.w3.org/2005/xpath-functions"
                exclude-result-prefixes="xs fn">

    <!-- Define the transformation name and purpose -->
    <xsl:comment>DocPackResponseToEnvelope: Maps DocPack response to envelope</xsl:comment>

    <!-- Define the input and output schema -->
    <xsl:comment>Source Schema: {}</xsl:comment>
    <xsl:comment>Target Schema: {}</xsl:comment>

    <!-- Define the field mapping requirements -->
    <xsl:comment>Field Mapping Requirements: []</xsl:comment>

    <!-- Define the business rules -->
    <xsl:comment>Business Rules: []</xsl:comment>

    <!-- Define the transformation patterns -->
    <xsl:comment>Transformation Patterns: []</xsl:comment>

    <!-- Define the template to match the root element of the source document -->
    <xsl:template match="/">
        <!-- Create the root element of the target document -->
        <Envelope>
            <!-- Apply the field mapping requirements -->
            <xsl:apply-templates select="*"/>
        </Envelope>
    </xsl:template>

    <!-- Define the template to match each element in the source document -->
    <xsl:template match="*">
        <!-- Check if the element has any child elements or text content -->
        <xsl:choose>
            <xsl:when test="* or text()">
                <!-- Create an element with the same name as the current element -->
                <xsl:element name="{name()}">
                    <!-- Recursively apply the templates to the child elements -->
                    <xsl:apply-templates select="*"/>
                </xsl:element>
            </xsl:when>
            <xsl:otherwise>
                <!-- Create an element with the same name as the current element and set its text content -->
                <xsl:element name="{name()}">
                    <xsl:value-of select="."/>
                </xsl:element>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Define the template to handle missing or invalid data -->
    <xsl:template match="@* | node()">
        <xsl:comment>Handling missing or invalid data</xsl:comment>
        <xsl:choose>
            <xsl:when test="not(.)">
                <!-- Handle missing data -->
                <xsl:element name="{name()}">
                    <xsl:text>Missing data</xsl:text>
                </xsl:element>
            </xsl:when>
            <xsl:otherwise>
                <!-- Handle invalid data -->
                <xsl:element name="{name()}">
                    <xsl:text>Invalid data</xsl:text>
                </xsl:element>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

</xsl:stylesheet>
```