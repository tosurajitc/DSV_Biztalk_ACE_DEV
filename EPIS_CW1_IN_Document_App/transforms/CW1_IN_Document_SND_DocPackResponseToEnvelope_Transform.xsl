```xsl
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema"
                xmlns:fn="http://www.w3.org/2005/xpath-functions"
                exclude-result-prefixes="xs fn"
                xmlns:doc="http://vector-db.com/doc"
                xmlns:env="http://vector-db.com/env">

    <!-- Define the output method and encoding -->
    <xsl:output method="xml" encoding="UTF-8" indent="yes"/>

    <!-- Define the template to match the root element of the source document -->
    <xsl:template match="/">
        <!-- Create the root element of the target document -->
        <env:Envelope>
            <!-- Apply templates to child elements of the source document -->
            <xsl:apply-templates select="*"/>
        </env:Envelope>
    </xsl:template>

    <!-- Define templates for each element in the source document -->
    <xsl:template match="*">
        <!-- Create an element in the target document with the same name as the source element -->
        <xsl:element name="{local-name()}" namespace="http://vector-db.com/env">
            <!-- Apply templates to child elements of the current element -->
            <xsl:apply-templates select="*"/>
        </xsl:element>
    </xsl:template>

    <!-- Define a template to handle missing or invalid data -->
    <xsl:template match="*[not(text())]">
        <!-- Create an element in the target document with a default value -->
        <xsl:element name="{local-name()}" namespace="http://vector-db.com/env">
            <xsl:text>Unknown</xsl:text>
        </xsl:element>
    </xsl:template>

    <!-- Define a template to handle error handling for missing or invalid data -->
    <xsl:template match="*[not(text()) and @required='true']">
        <!-- Create an error element in the target document -->
        <env:Error>
            <env:Message>Missing required field: <xsl:value-of select="local-name()"/></env:Message>
        </env:Error>
    </xsl:template>

    <!-- Define a template to handle business rule validation -->
    <xsl:template match="*[starts-with(local-name(), 'validate-')]">
        <!-- Create a validation element in the target document -->
        <env:Validation>
            <env:Rule><xsl:value-of select="local-name()"/></env:Rule>
            <env:Result>
                <xsl:choose>
                    <xsl:when test=". = 'true'">Pass</xsl:when>
                    <xsl:otherwise>Fail</xsl:otherwise>
                </xsl:choose>
            </env:Result>
        </env:Validation>
    </xsl:template>

    <!-- Define a template to handle conditional logic -->
    <xsl:template match="*[starts-with(local-name(), 'if-')]">
        <!-- Create a conditional element in the target document -->
        <env:Conditional>
            <env:Test><xsl:value-of select="local-name()"/></env:Test>
            <env:Result>
                <xsl:choose>
                    <xsl:when test=". = 'true'">True</xsl:when>
                    <xsl:otherwise>False</xsl:otherwise>
                </xsl:choose>
            </env:Result>
        </env:Conditional>
    </xsl:template>

</xsl:stylesheet>
```